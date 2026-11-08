"""Stable scoring API for Part 3: PIL images -> P(fake) using a saved head.

No format conversion happens here; images are scored exactly as given. Before scoring, the
feature-extraction settings recorded in the head are checked against the config and the backbone
in use, so a head is never applied to features made with other settings.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from common import load_config
from part2.backbones import Backbone, load_backbone
from part2.head import load_head, predict_fake
from part2.preprocess import apply_pipeline, check_config_matches_code, to_tensor

# Repo root: this file lives in scripts/part2/.
REPO_ROOT = Path(__file__).resolve().parents[2]
# Head-meta fields scoring relies on. Heads written before part2_run.py recorded the
# feature-extraction settings lack some of them and must be regenerated.
REQUIRED_HEAD_FIELDS = ("backbone", "dim", "pipeline", "backbone_cfg", "pipeline_cfg", "normalisation")


def embed_images(images: Sequence[Image.Image], backbone: Backbone, pipeline: str,
                 batch_size: int = 64) -> np.ndarray:
    """Return an (N, dim) float32 matrix of L2-normalised embeddings.

    ImageTooSmall (pipeline B) propagates to the caller.
    """
    import torch

    chunks = []
    for start in range(0, len(images), batch_size):
        tensors = [
            to_tensor(apply_pipeline(img.convert("RGB"), pipeline), backbone.mean, backbone.std)
            for img in images[start:start + batch_size]
        ]
        chunks.append(backbone.encode(torch.stack(tensors)))
    if not chunks:
        return np.zeros((0, backbone.dim), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def _same_floats(a: Sequence[float], b: Sequence[float]) -> bool:
    """Equal float sequences (JSON round-trips floats exactly; this only ignores tuple vs list)."""
    a, b = [float(x) for x in a], [float(x) for x in b]
    return len(a) == len(b) and bool(np.allclose(a, b, rtol=0.0, atol=1e-12))


def check_head_settings(meta: dict, cfg: dict, head_path: Path | str = "head") -> None:
    """Raise ValueError unless the head records its feature-extraction settings and they match ``cfg``."""
    missing = [k for k in REQUIRED_HEAD_FIELDS if k not in meta]
    if missing:
        raise ValueError(
            f"head {head_path} does not record its feature-extraction settings (missing {missing}); it was "
            "written by an older part2_run.py. Re-run part2_run.py with --overwrite to regenerate it.")
    name, pipeline = meta["backbone"], meta["pipeline"]
    configured = (cfg.get("backbones") or {}).get(name)
    if configured != meta["backbone_cfg"]:
        raise ValueError(f"head {head_path} was trained on {name!r} features from {meta['backbone_cfg']}, "
                         f"but the config defines {name!r} as {configured}")
    configured = (cfg.get("pipelines") or {}).get(pipeline)
    if configured != meta["pipeline_cfg"]:
        raise ValueError(f"head {head_path} was trained with pipeline {pipeline!r} = {meta['pipeline_cfg']}, "
                         f"but the config defines it as {configured}")


def check_backbone(meta: dict, backbone: Backbone, head_path: Path | str = "head") -> None:
    """Raise ValueError unless ``backbone`` matches the head's name, dim, normalisation and config block."""
    problems = []
    if backbone.name != meta["backbone"]:
        problems.append(f"name {backbone.name!r} != {meta['backbone']!r}")
    if backbone.dim != int(meta["dim"]):
        problems.append(f"dim {backbone.dim} != {meta['dim']}")
    norm = meta["normalisation"]
    if not (_same_floats(backbone.mean, norm["mean"]) and _same_floats(backbone.std, norm["std"])):
        problems.append(f"normalisation mean/std {backbone.mean}/{backbone.std} != {norm['mean']}/{norm['std']}")
    if backbone.config is not None and backbone.config != meta["backbone_cfg"]:
        problems.append(f"config {backbone.config} != {meta['backbone_cfg']}")
    if problems:
        raise ValueError(f"backbone does not match head {head_path}: " + "; ".join(problems))


def score_images(images: Sequence[Image.Image], head_path: Path, backbone: Backbone | None = None,
                 batch_size: int = 64, cfg: dict | None = None) -> np.ndarray:
    """Return P(fake) per image using the backbone and pipeline recorded in the head's meta.

    ``cfg`` defaults to the repo's config/part2.yaml. The head's backbone and pipeline settings must
    equal the config's, and the backbone (given, or loaded from the config) must match the head's
    name, dim and normalisation; otherwise ValueError. Load the backbone once with load_backbone
    and pass it in when scoring many batches.
    """
    head = load_head(head_path)
    meta = head.meta
    if cfg is None:
        cfg = load_config(REPO_ROOT, "config/part2.yaml")
    check_config_matches_code(cfg)
    check_head_settings(meta, cfg, head_path)  # before loading weights: fail fast on a mismatch
    if backbone is None:
        backbone = load_backbone(meta["backbone"], cfg, REPO_ROOT)
    check_backbone(meta, backbone, head_path)
    return predict_fake(head, embed_images(images, backbone, meta["pipeline"], batch_size))
