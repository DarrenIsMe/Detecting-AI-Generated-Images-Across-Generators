"""Part 3 robustness run: perturb test images and score them with the Part 2 heads.

Steps (each one is a function below):
1. load_heads      - read the Part 3 config and check each head against config/part2.yaml, so a head
                     is never applied to features made with other settings.
2. build_sample    - rebuild the protocol's test groups exactly as Part 2 does (same small-image
                     exclusion, same leakage check), then draw a seeded, class-balanced sample.
3. embed_condition - for one perturbation, stream the sampled images once: decode, re-encode fakes
                     for matched_jpeg heads, perturb, embed with each pipeline the heads need.
                     Saved per perturbation, so an interrupted run resumes where it stopped.
4. score           - apply every head to the matching embeddings and write one per-image file.

Protocols:
- strict (config/part3.yaml): four heads trained on BigGAN; every head scores every test group
  (ADM/GLIDE/SD15/VQDM/Wukong, plus Midjourney reported separately).
- logo (config/part3_logo.yaml): one head per held-out generator and condition (28 heads); each
  scores only its own fold's test group. The same real images sit in all seven folds' test sets,
  so every image is embedded once and shared between the folds' heads.

Only test images are ever transformed; nothing here trains or changes a head.

Usage (repo root):
    python scripts/part3_run.py --per-class 10                            # quick check (strict)
    python scripts/part3_run.py                                           # full strict run
    python scripts/part3_run.py --config config/part3_logo.yaml           # full LOGO run
    python scripts/part3_tables.py [--config config/part3_logo.yaml]      # tables and figure
    python scripts/part3_analysis.py [--config config/part3_logo.yaml]    # failure cases, t-SNE, ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Silence a harmless Hugging Face symlink warning on Windows (as part2_predict.py does).
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import pandas as pd

from common import load_config, write_csv, write_json
from part2.backbones import Backbone, load_backbone
from part2.data import iter_image_bytes, load_manifest, small_image_ids
from part2.formats import JpegSpec, load_condition_image
from part2.head import FittedHead, load_head, predict_fake
from part2.preprocess import B_CROP, apply_pipeline, check_config_matches_code, to_tensor
from part2.protocols import ProtocolSplit, assert_no_leakage, load_protocol
from part2.scoring import check_backbone, check_head_settings
from part3.perturb import Perturbation, parse_perturbations
from part3.sampling import sample_group

ROOT = Path(__file__).resolve().parents[1]
# Bump when the embedding code changes, so old caches are recomputed instead of reused.
CACHE_VERSION = 1
SAMPLE_COLUMNS = ["image_id", "evaluation_group", "in_overall", "label", "generator"]
PROTOCOLS = ("strict", "logo")


# ---------------------------------------------------------------- 1. heads


@dataclass
class HeadInfo:
    """A Part 2 head plus the test condition it expects."""

    name: str              # file stem, e.g. clip_l14__A__raw__strict__strict__full
    head: FittedHead
    pipeline: str          # "A" or "B"
    fmt: str               # "raw" or "matched_jpeg"
    jpeg: JpegSpec | None  # recipe used to re-encode fakes (matched_jpeg only)
    group: str | None      # None: scores every test group (strict); else only this LOGO fold

    @property
    def combo(self) -> tuple[str, str]:
        return self.pipeline, self.fmt


def load_heads(root: Path, cfg2: dict, cfg3: dict, folds: list[str]) -> list[HeadInfo]:
    """Load and validate every head listed in the config.

    For LOGO, each listed path contains ``{fold}`` and is expanded to one head per fold.
    """
    protocol = cfg3["protocol"]
    heads = []
    for entry in cfg3["heads"]:
        templated = "{fold}" in entry
        if templated != (protocol == "logo"):
            raise ValueError(f"{entry}: LOGO head paths need a {{fold}} placeholder; strict paths must not have one")
        targets = [(entry.format(fold=f), f) for f in folds] if templated else [(entry, None)]
        for rel, fold in targets:
            path = root / rel
            head = load_head(path)
            meta = head.meta
            check_head_settings(meta, cfg2, path)  # Part 2's own check: backbone/pipeline settings match
            if meta["backbone"] != cfg3["backbone"]:
                raise ValueError(f"{rel}: backbone {meta['backbone']!r}, config expects {cfg3['backbone']!r}")
            # A head from another protocol may have trained on these test generators.
            if meta["protocol"] != protocol:
                raise ValueError(f"{rel}: trained on protocol {meta['protocol']!r}, but Part 3 tests "
                                 f"{protocol!r} test rows")
            if fold is not None and meta.get("fold") != fold:
                raise ValueError(f"{rel}: head was trained for fold {meta.get('fold')!r}, not {fold!r}")
            jpeg = None
            if meta["format"] == "matched_jpeg":
                if not meta.get("jpeg"):
                    raise ValueError(f"{rel}: matched_jpeg head without a recorded JPEG recipe")
                jpeg = JpegSpec(int(meta["jpeg"]["quality"]), str(meta["jpeg"]["subsampling"]))
            heads.append(HeadInfo(path.name.removesuffix(".json"), head, meta["pipeline"], meta["format"],
                                  jpeg, fold))
    if not heads:
        raise ValueError("the config lists no heads")
    return heads


def jpeg_by_format(heads: list[HeadInfo]) -> dict[str, JpegSpec | None]:
    """One re-encoding recipe per format. All matched_jpeg heads must agree, because the
    re-encoded fakes are embedded once and shared between them."""
    specs: dict[str, JpegSpec | None] = {}
    for h in heads:
        if h.fmt in specs and specs[h.fmt] != h.jpeg:
            raise ValueError(f"heads disagree on the {h.fmt} JPEG recipe: {specs[h.fmt]} vs {h.jpeg}")
        specs[h.fmt] = h.jpeg
    return specs


# ---------------------------------------------------------------- 2. sample


def protocol_splits(root: Path, cfg2: dict, protocol: str, manifest: list[dict]) -> list[ProtocolSplit]:
    """The protocol's splits, built exactly as Part 2 builds them."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}, got {protocol!r}")
    # Same exclusion as Part 2: images under min_side were never in any Part 2 test set.
    excluded = frozenset(small_image_ids(manifest, cfg2["min_side"]))
    splits = load_protocol(protocol, root / "data" / "splits", exclude_ids=excluded)
    for split in splits:
        assert_no_leakage(split)
    return splits


def build_sample(splits: list[ProtocolSplit], by_id: dict[str, dict], per_class: int | None,
                 seed: int) -> tuple[list[dict], list[dict]]:
    """Seeded sample of test rows, as (members, images).

    members: one row per (evaluation group, image), i.e. what gets scored and evaluated.
    images:  each sampled image once, with its shard location, i.e. what gets embedded.
    Under strict the two have the same images; under LOGO a real image can belong to several folds.
    """
    members = []
    for split in splits:
        for groups, in_overall in ((split.test_groups, True), (split.separate_groups, False)):
            for group in sorted(groups):
                for row in sample_group(groups[group], per_class, seed, group):
                    members.append({"image_id": row["image_id"], "evaluation_group": group,
                                    "in_overall": in_overall, "label": int(row["label"]),
                                    "generator": row["generator"]})
    images: dict[str, dict] = {}
    for m in members:
        source = by_id.get(m["image_id"])
        if source is None:
            raise KeyError(f"{m['image_id']} is in the split but not the manifest; "
                           "regenerate it with scripts/inspect_datasets.py")
        if int(source["label"]) != m["label"]:
            raise ValueError(f"{m['image_id']}: label differs between split and manifest")
        # where iter_image_bytes finds the encoded bytes
        images.setdefault(m["image_id"], {"image_id": m["image_id"], "label": m["label"],
                                          "shard": source["shard"], "row_in_shard": source["row_in_shard"]})
    return members, sorted(images.values(), key=lambda r: r["image_id"])


# ---------------------------------------------------------------- 3. embeddings


def cache_signature(pert: Perturbation, ids: list[str], combos: list[tuple[str, str]],
                    jpeg: dict[str, JpegSpec | None], cfg2: dict, backbone_name: str) -> str:
    """Hash of everything that determines a perturbation's embeddings. A cache is reused only
    when this matches, so changing the sample, a setting or the code version recomputes it."""
    payload = {
        "version": CACHE_VERSION, "perturbation": pert.spec(), "ids": ids, "combos": combos,
        "jpeg": {k: (None if v is None else [v.quality, v.subsampling]) for k, v in sorted(jpeg.items())},
        "backbone": cfg2["backbones"][backbone_name], "pipelines": cfg2["pipelines"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def combo_key(combo: tuple[str, str]) -> str:
    return f"emb__{combo[0]}__{combo[1]}"


def load_cache(path: Path, signature: str, combos: list[tuple[str, str]]) -> dict | None:
    """The cached embeddings when they exist and match ``signature``; otherwise None."""
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["signature"]) != signature:
                return None
            return {combo: archive[combo_key(combo)] for combo in combos}
    except (OSError, KeyError, ValueError):  # unreadable or partial file: recompute
        return None


def save_cache(path: Path, signature: str, ids: list[str], embeddings: dict) -> None:
    """Write atomically (temporary file, then rename), so a crash never leaves a half file
    that looks complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")  # np.savez keeps a name ending in .npz
    np.savez(tmp, signature=np.array(signature), ids=np.array(ids),
             **{combo_key(c): e for c, e in embeddings.items()})
    os.replace(tmp, path)


def effective_format(fmt: str, label: int) -> str:
    """matched_jpeg only re-encodes fakes; a real image is the same under both formats, so it is
    perturbed and embedded once and shared (saves about a quarter of the GPU time)."""
    return "matched_jpeg" if fmt == "matched_jpeg" and label == 1 else "raw"


def embed_condition(pert: Perturbation, images: list[dict], combos: list[tuple[str, str]],
                    jpeg: dict[str, JpegSpec | None], backbone: Backbone, root: Path,
                    batch_size: int, workers: int = 1) -> dict:
    """Embeddings of every sampled image under one perturbation, for every (pipeline, format).

    Returns {(pipeline, format): float32 array (n_images, dim)}. A row is NaN when the perturbed
    image is too small for that pipeline (pipeline B crops 112 native pixels), so it is skipped
    and counted rather than crashing the run.
    """
    import torch

    index = {row["image_id"]: i for i, row in enumerate(images)}
    out = {c: np.full((len(images), backbone.dim), np.nan, dtype=np.float32) for c in combos}
    formats = sorted({fmt for _, fmt in combos})
    pipelines = sorted({pipe for pipe, _ in combos})

    def prepare(item: tuple[int, int, bytes]) -> list[tuple]:
        """Everything before the model for one image; runs in a worker thread.

        Returns (pipeline, row index, label, effective format, tensor) per distinct variant.
        Decoding, JPEG re-encoding, perturbing and resizing are CPU work, so running several
        images at once (as Part 2's extraction does with decode_threads) keeps the GPU fed.
        """
        i, label, data = item
        variants = []
        for eff in sorted({effective_format(f, label) for f in formats}):
            image = pert.apply(load_condition_image(data, label, eff, jpeg.get(eff)))  # Part 2's loader
            for pipe in pipelines:
                # Same rule as Part 2's pipeline B crop: the shorter side must be at least 112 px.
                if pipe == "B" and min(image.size) < B_CROP:
                    continue
                # Exactly what Part 2's embed_images does to each image before the model.
                tensor = to_tensor(apply_pipeline(image.convert("RGB"), pipe), backbone.mean, backbone.std)
                variants.append((pipe, i, label, eff, tensor))
        return variants

    def flush(batch: list[tuple[int, int, bytes]], pool: ThreadPoolExecutor) -> None:
        prepared = [v for variants in pool.map(prepare, batch) for v in variants]  # map keeps input order
        for pipe in pipelines:
            rows = [v for v in prepared if v[0] == pipe]
            # Same order and same chunks of batch_size as embed_images, so the embeddings are identical.
            for start in range(0, len(rows), batch_size):
                chunk = rows[start:start + batch_size]
                emb = backbone.encode(torch.stack([v[4] for v in chunk]))
                for (_, i, label, eff, _), vector in zip(chunk, emb):
                    for fmt in formats:
                        if (pipe, fmt) in out and effective_format(fmt, label) == eff:
                            out[(pipe, fmt)][i] = vector

    batch: list[tuple[int, int, bytes]] = []
    done, start = 0, time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for row, data in iter_image_bytes(images, root):  # shard order; rows are matched by image_id
            batch.append((index[row["image_id"]], int(row["label"]), data))
            if len(batch) >= batch_size:
                flush(batch, pool)
                done += len(batch)
                batch = []
                if done % (batch_size * 20) == 0:
                    rate = done / max(time.time() - start, 1e-9)
                    print(f"  {pert.name}: {done}/{len(images)} images ({rate:.1f}/s)", flush=True)
        if batch:
            flush(batch, pool)
    return out


# ---------------------------------------------------------------- 4. scoring


def score(heads: list[HeadInfo], pert: Perturbation, embeddings: dict, images: list[dict],
          members: list[dict]) -> pd.DataFrame:
    """Per-row P(fake) under one perturbation; each head scores only the groups it applies to.
    p_fake is NaN where the image was too small to embed."""
    index = {row["image_id"]: i for i, row in enumerate(images)}
    base = pd.DataFrame(members, columns=SAMPLE_COLUMNS)
    rows_of = base["image_id"].map(index).to_numpy()
    frames = []
    for h in heads:
        mask = np.ones(len(base), bool) if h.group is None else (base["evaluation_group"] == h.group).to_numpy()
        X = embeddings[h.combo][rows_of[mask]]
        scored = ~np.isnan(X).any(axis=1)
        p_fake = np.full(len(X), np.nan)
        if scored.any():
            p_fake[scored] = predict_fake(h.head, X[scored])
        frame = base[mask].reset_index(drop=True)
        frame.insert(0, "perturbation", pert.name)
        frame.insert(0, "format", h.fmt)
        frame.insert(0, "pipeline", h.pipeline)
        frame.insert(0, "head", h.name)
        frame["scored"] = scored
        frame["p_fake"] = p_fake
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/part3.yaml")
    parser.add_argument("--per-class", type=int, default=None,
                        help="override sample.per_class_per_group (e.g. 10 for a quick check)")
    parser.add_argument("--only", nargs="+", default=None,
                        help="compute only these perturbations now (others are scored from cache if present)")
    parser.add_argument("--overwrite", action="store_true", help="recompute embeddings even if cached")
    parser.add_argument("--device", default="auto", help="auto, cuda or cpu")
    args = parser.parse_args(argv)

    cfg2 = load_config(ROOT, "config/part2.yaml")
    cfg3 = load_config(ROOT, args.config)
    check_config_matches_code(cfg2)  # Part 2's guard: config and hard-coded pipelines agree
    protocol = cfg3["protocol"]
    local, results = ROOT / cfg3["local_root"], ROOT / cfg3["results_root"]
    local.mkdir(parents=True, exist_ok=True)
    # Keep embeddings and predictions out of Git without editing the shared .gitignore.
    (local / ".gitignore").write_text("*\n", encoding="utf-8")

    perts = parse_perturbations(cfg3["perturbations"])
    names = [p.name for p in perts]
    if args.only and (unknown := set(args.only) - set(names)):
        raise SystemExit(f"--only: unknown perturbations {sorted(unknown)}; choose from {names}")

    manifest = load_manifest(ROOT)
    if not manifest:
        raise SystemExit("data/manifests/master_manifest.csv is missing or empty; "
                         "run `python scripts/inspect_datasets.py` first")
    splits = protocol_splits(ROOT, cfg2, protocol, manifest)
    heads = load_heads(ROOT, cfg2, cfg3, [s.fold for s in splits] if protocol == "logo" else [])
    combos = sorted({h.combo for h in heads})
    jpeg = jpeg_by_format(heads)
    per_class = args.per_class if args.per_class is not None else cfg3["sample"]["per_class_per_group"]
    members, images = build_sample(splits, {r["image_id"]: r for r in manifest}, per_class,
                                   int(cfg3["sample"]["seed"]))
    ids = [row["image_id"] for row in images]
    print(f"{protocol}: {len(images)} test images ({len(members)} group rows), {len(heads)} heads, "
          f"{len(perts)} perturbations, combos {combos}")

    backbone: Backbone | None = None
    frames, meta_rows = [], []
    for pert in perts:
        path = local / "embeddings" / f"{cfg3['backbone']}__{pert.name}.npz"
        signature = cache_signature(pert, ids, combos, jpeg, cfg2, cfg3["backbone"])
        embeddings = None if args.overwrite else load_cache(path, signature, combos)
        wanted = args.only is None or pert.name in args.only
        if embeddings is None and wanted:
            if backbone is None:  # slow: load once, only when something needs computing
                backbone = load_backbone(cfg3["backbone"], cfg2, ROOT, device=args.device)
                for h in heads:
                    check_backbone(h.head.meta, backbone, h.name)
                print(f"backbone {backbone.name} on {backbone.device}")
            t0 = time.time()
            # Same number of image-preparation threads as Part 2's extraction (decode_threads).
            embeddings = embed_condition(pert, images, combos, jpeg, backbone, ROOT, int(cfg3["batch_size"]),
                                         workers=int(cfg2["extraction"]["decode_threads"]))
            save_cache(path, signature, ids, embeddings)
            print(f"{pert.name}: embedded in {time.time() - t0:.0f}s")
        elif embeddings is None:
            print(f"{pert.name}: not cached and not in --only; left out of the predictions")
            continue
        else:
            print(f"{pert.name}: reusing cached embeddings")
        frames.append(score(heads, pert, embeddings, images, members))
        meta_rows.append({**pert.spec(), "signature": signature, "skipped_images_per_combo": {
            f"{p}/{f}": int(np.isnan(e).any(axis=1).sum()) for (p, f), e in embeddings.items()}})

    if not frames:
        raise SystemExit("no perturbation has embeddings yet; run without --only first")
    predictions = pd.concat(frames, ignore_index=True)
    out = local / "predictions" / f"{cfg3['backbone']}__{protocol}.csv.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out, index=False)

    # Small, committed records: exactly which images were tested and with what settings.
    write_csv(results / "sample.csv", members, SAMPLE_COLUMNS)
    write_json(results / "run_meta.json", {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": args.config, "protocol": protocol, "backbone": cfg3["backbone"],
        "per_class_per_group": per_class, "seed": cfg3["sample"]["seed"],
        "n_images": len(images), "n_group_rows": len(members), "heads": [h.name for h in heads],
        "jpeg_recipe": {k: (None if v is None else {"quality": v.quality, "subsampling": v.subsampling})
                        for k, v in jpeg.items()},
        "perturbations": meta_rows,
    })
    print(f"wrote {out.relative_to(ROOT)} ({len(predictions)} rows) and {results.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
