"""Frozen-feature cache for Part 2: spec hashing, save/load, row lookup and extraction."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from typing import Callable, Iterable, Sequence

import numpy as np
import torch

from common import load_config
from part2.backbones import Backbone
from part2.formats import JpegSpec, load_condition_image
from part2.preprocess import MIN_SIDE, ImageTooSmall, apply_pipeline, to_tensor

# The experiment grid: every feature set is one (backbone, pipeline, format). Backbones are in run
# order (clip_l14 first: the single-image interface needs it). part2_extract.py and part2_run.py
# both import these, so the two CLIs cannot drift apart.
BACKBONE_ORDER = ("clip_l14", "clip_b32", "resnet50")
PIPELINES = ("A", "B")
FORMATS = ("raw", "matched_jpeg")

# Reason recorded for images pipeline B cannot crop ("min_side<112", tied to the crop size).
SKIP_REASON = f"min_side<{MIN_SIDE}"


class StaleFeatureCache(RuntimeError):
    """Raised when a cached feature set was built from a different spec."""


def feature_set_id(backbone: str, pipeline: str, fmt: str) -> str:
    """Directory name of one cached feature set."""
    return f"{backbone}__{pipeline}__{fmt}"


def feature_spec(
    backbone: str,
    pipeline: str,
    fmt: str,
    cfg: dict,
    jpeg: "JpegSpec | None",
    image_ids: list[str],
) -> dict:
    """Everything that determines the feature values, as a JSON-able dict."""
    # Hash of the ordered ids keeps the spec small but sensitive to the image set.
    ids_hash = hashlib.sha256("\n".join(image_ids).encode("utf-8")).hexdigest()
    return {
        "backbone": backbone,
        "backbone_cfg": cfg["backbones"][backbone],
        "pipeline": pipeline,
        "pipeline_cfg": cfg["pipelines"][pipeline],
        "format": fmt,
        "jpeg": dataclasses.asdict(jpeg) if jpeg is not None else None,
        "image_ids_sha256": ids_hash,
    }


def spec_hash(spec: dict) -> str:
    """sha256 of the canonical (sorted, compact) JSON of a spec."""
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class FeatureSet:
    """Cached embeddings with an image_id -> row index lookup."""

    ids: np.ndarray
    X: np.ndarray
    meta: dict
    index: dict[str, int]

    def rows(self, image_ids: Sequence[str]) -> np.ndarray:
        """Embeddings for ``image_ids`` in the given order; KeyError names missing ids."""
        missing = [i for i in image_ids if i not in self.index]
        if missing:
            shown = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
            raise KeyError(f"{len(missing)} image ids missing from feature set: {shown}")
        return self.X[[self.index[i] for i in image_ids]]


def _atomic_write(path: Path, writer: Callable) -> None:
    """Write via a temp file in the same directory, then replace."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        writer(handle)
    os.replace(tmp, path)


def save_feature_set(directory: Path, ids: list[str], X: np.ndarray, meta: dict) -> None:
    """Write ``features.npz`` and ``meta.json`` atomically."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_write(
        directory / "features.npz",
        lambda h: np.savez(h, image_ids=np.asarray(ids, dtype=str), embeddings=np.asarray(X)),
    )
    payload = json.dumps(meta, indent=2, sort_keys=True).encode("utf-8")
    _atomic_write(directory / "meta.json", lambda h: h.write(payload))


def load_feature_set_dir(directory: Path, expected_spec_hash: str | None = None) -> FeatureSet:
    """Load a cache directory; reject it if its spec hash differs from the expected one."""
    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    if expected_spec_hash is not None and meta.get("spec_hash") != expected_spec_hash:
        raise StaleFeatureCache(
            f"{directory}: cached spec_hash {meta.get('spec_hash')!r} != expected {expected_spec_hash!r}"
        )
    with np.load(directory / "features.npz", allow_pickle=False) as data:
        ids = data["image_ids"]
        X = data["embeddings"]
    index = {str(i): n for n, i in enumerate(ids)}
    return FeatureSet(ids=ids, X=X, meta=meta, index=index)


def load_feature_set(
    root: Path,
    backbone: str,
    pipeline: str,
    fmt: str,
    expected_spec_hash: str | None = None,
) -> FeatureSet:
    """Load ``<root>/<local_root>/features/<feature_set_id>/``."""
    cfg = load_config(root, "config/part2.yaml")
    directory = Path(root) / cfg["local_root"] / "features" / feature_set_id(backbone, pipeline, fmt)
    return load_feature_set_dir(directory, expected_spec_hash)


# torch.cuda.OutOfMemoryError exists on all supported torch builds; fall back defensively.
_OOM = getattr(torch.cuda, "OutOfMemoryError", MemoryError)


def _encode_with_oom_retry(backbone: Backbone, batch: torch.Tensor, state: dict) -> np.ndarray:
    """Encode ``batch``; on CUDA OOM halve ``state['batch_size']`` and retry in smaller pieces."""
    while True:
        size = max(1, min(state["batch_size"], len(batch)))
        try:
            parts = [backbone.encode(batch[i:i + size]) for i in range(0, len(batch), size)]
            return np.concatenate(parts, axis=0)
        except _OOM:
            if size == 1:
                raise
            state["batch_size"] = max(1, size // 2)
            torch.cuda.empty_cache()
            print(f"CUDA OOM: retrying with batch size {state['batch_size']}", flush=True)


def extract_features(
    items: Iterable[tuple[dict, bytes]],
    backbone: Backbone,
    pipeline: str,
    fmt: str,
    jpeg: JpegSpec | None,
    batch_size: int = 64,
    decode_threads: int = 1,
) -> tuple[list[str], np.ndarray, list[dict]]:
    """Embed ``(manifest_row, encoded_bytes)`` items; returns ids (input order), X, skipped.

    Under pipeline B, images with a side shorter than the crop are skipped and reported.
    Decode errors propagate.
    """
    def prepare(item):
        # Decode (+ matched-JPEG re-encode), run the pipeline, normalise; None means too small.
        row, data = item
        image = load_condition_image(data, int(row["label"]), fmt, jpeg)
        try:
            image = apply_pipeline(image, pipeline)
        except ImageTooSmall:
            return row["image_id"], None
        return row["image_id"], to_tensor(image, backbone.mean, backbone.std)

    ids: list[str] = []
    skipped: list[dict] = []
    chunks: list[np.ndarray] = []
    state = {"batch_size": batch_size}
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=max(1, decode_threads)) as pool:
        while True:
            chunk = list(islice(iterator, state["batch_size"]))
            if not chunk:
                break
            kept = []
            for image_id, tensor in pool.map(prepare, chunk):  # map preserves input order
                if tensor is None:
                    skipped.append({"image_id": image_id, "reason": SKIP_REASON})
                else:
                    ids.append(image_id)
                    kept.append(tensor)
            if kept:
                chunks.append(_encode_with_oom_retry(backbone, torch.stack(kept), state))
    X = np.concatenate(chunks, axis=0).astype(np.float32) if chunks else np.zeros((0, backbone.dim), np.float32)
    return ids, X, skipped
