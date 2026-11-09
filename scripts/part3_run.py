"""Part 3 robustness run: perturb strict test images and score them with every Part 2 head.

Steps (each one is a function below):
1. load_heads     - read config/part3.yaml and check each head against config/part2.yaml, so a head
                    is never applied to features made with other settings.
2. build_sample   - rebuild the strict test groups exactly as Part 2 does (same small-image
                    exclusion, same leakage check), then draw a seeded, class-balanced sample.
3. embed_condition - for one perturbation, stream the sampled images once: decode, re-encode fakes
                    for matched_jpeg heads, perturb, embed with each pipeline the heads need.
                    Saved per perturbation, so an interrupted run resumes where it stopped.
4. score          - apply every head to the matching embeddings and write one per-image file.

Only test images are ever transformed; nothing here trains or changes a head.

Usage (repo root):
    python scripts/part3_run.py --per-class 10     # quick check that everything is wired up
    python scripts/part3_run.py                    # full run with config/part3.yaml
    python scripts/part3_tables.py                 # tables and figures from the predictions
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
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
from part2.preprocess import B_CROP, check_config_matches_code
from part2.protocols import assert_no_leakage, load_protocol
from part2.scoring import check_backbone, check_head_settings, embed_images
from part3.perturb import Perturbation, parse_perturbations
from part3.sampling import sample_group

ROOT = Path(__file__).resolve().parents[1]
# Bump when the embedding code changes, so old caches are recomputed instead of reused.
CACHE_VERSION = 1
SAMPLE_COLUMNS = ["image_id", "evaluation_group", "in_overall", "label", "generator"]


# ---------------------------------------------------------------- 1. heads


@dataclass
class HeadInfo:
    """A Part 2 head plus the test condition it expects."""

    name: str              # file stem, e.g. clip_l14__A__raw__strict__strict__full
    head: FittedHead
    pipeline: str          # "A" or "B"
    fmt: str               # "raw" or "matched_jpeg"
    jpeg: JpegSpec | None  # recipe used to re-encode fakes (matched_jpeg only)

    @property
    def combo(self) -> tuple[str, str]:
        return self.pipeline, self.fmt


def load_heads(root: Path, cfg2: dict, cfg3: dict) -> list[HeadInfo]:
    """Load and validate every head listed in config/part3.yaml."""
    heads = []
    for rel in cfg3["heads"]:
        path = root / rel
        head = load_head(path)
        meta = head.meta
        check_head_settings(meta, cfg2, path)  # Part 2's own check: backbone/pipeline settings match
        if meta["backbone"] != cfg3["backbone"]:
            raise ValueError(f"{rel}: backbone {meta['backbone']!r}, config expects {cfg3['backbone']!r}")
        # A head from another protocol (e.g. LOGO) may have trained on these test generators.
        if meta["protocol"] != cfg3["protocol"]:
            raise ValueError(f"{rel}: trained on protocol {meta['protocol']!r}, but Part 3 tests "
                             f"{cfg3['protocol']!r} test rows")
        jpeg = None
        if meta["format"] == "matched_jpeg":
            if not meta.get("jpeg"):
                raise ValueError(f"{rel}: matched_jpeg head without a recorded JPEG recipe")
            jpeg = JpegSpec(int(meta["jpeg"]["quality"]), str(meta["jpeg"]["subsampling"]))
        heads.append(HeadInfo(path.name.removesuffix(".json"), head, meta["pipeline"], meta["format"], jpeg))
    if not heads:
        raise ValueError("config/part3.yaml lists no heads")
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


def build_sample(root: Path, cfg2: dict, cfg3: dict, per_class: int | None) -> list[dict]:
    """Seeded sample of strict test rows, joined to the master manifest for image locations."""
    if cfg3["protocol"] != "strict":
        raise ValueError("only the strict protocol is supported (its test groups are fixed)")
    manifest = load_manifest(root)
    if not manifest:
        raise FileNotFoundError("data/manifests/master_manifest.csv is missing or empty; "
                                "run `python scripts/inspect_datasets.py` first")
    by_id = {row["image_id"]: row for row in manifest}
    # Same exclusion as Part 2: images under min_side were never in any Part 2 test set.
    excluded = frozenset(small_image_ids(manifest, cfg2["min_side"]))
    [split] = load_protocol("strict", root / "data" / "splits", exclude_ids=excluded)
    assert_no_leakage(split)

    seed = int(cfg3["sample"]["seed"])
    sample = []
    for groups, in_overall in ((split.test_groups, True), (split.separate_groups, False)):
        for group in sorted(groups):
            for row in sample_group(groups[group], per_class, seed, group):
                source = by_id.get(row["image_id"])
                if source is None:
                    raise KeyError(f"{row['image_id']} is in the split but not the manifest; "
                                   "regenerate it with scripts/inspect_datasets.py")
                if source["label"] != row["label"]:
                    raise ValueError(f"{row['image_id']}: label differs between split and manifest")
                sample.append({
                    "image_id": row["image_id"], "evaluation_group": group, "in_overall": in_overall,
                    "label": int(row["label"]), "generator": row["generator"],
                    # where iter_image_bytes finds the encoded bytes
                    "shard": source["shard"], "row_in_shard": source["row_in_shard"],
                })
    return sample


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


def embed_condition(pert: Perturbation, sample: list[dict], combos: list[tuple[str, str]],
                    jpeg: dict[str, JpegSpec | None], backbone: Backbone, root: Path,
                    batch_size: int) -> dict:
    """Embeddings of every sampled image under one perturbation, for every (pipeline, format).

    Returns {(pipeline, format): float32 array (n, dim)}. A row is NaN when the perturbed image
    is too small for that pipeline (pipeline B crops 112 native pixels), so it is skipped and
    counted rather than crashing the run.
    """
    index = {row["image_id"]: i for i, row in enumerate(sample)}
    out = {c: np.full((len(sample), backbone.dim), np.nan, dtype=np.float32) for c in combos}
    formats = sorted({fmt for _, fmt in combos})
    pipelines = sorted({pipe for pipe, _ in combos})

    def flush(batch: list[tuple[int, int, bytes]]) -> None:
        # (row index, effective format, perturbed image) for each distinct variant
        variants = []
        for i, label, data in batch:
            for eff in sorted({effective_format(f, label) for f in formats}):
                base = load_condition_image(data, label, eff, jpeg.get(eff))  # Part 2's own loader
                variants.append((i, label, eff, pert.apply(base)))
        for pipe in pipelines:
            usable = [v for v in variants if pipe != "B" or min(v[3].size) >= B_CROP]
            if not usable:
                continue
            emb = embed_images([v[3] for v in usable], backbone, pipe, batch_size)
            for (i, label, eff, _), vector in zip(usable, emb):
                for fmt in formats:
                    if (pipe, fmt) in out and effective_format(fmt, label) == eff:
                        out[(pipe, fmt)][i] = vector

    batch: list[tuple[int, int, bytes]] = []
    done, start = 0, time.time()
    for row, data in iter_image_bytes(sample, root):  # shard order; rows are matched by image_id
        batch.append((index[row["image_id"]], int(row["label"]), data))
        if len(batch) >= batch_size:
            flush(batch)
            done += len(batch)
            batch = []
            if done % (batch_size * 20) == 0:
                rate = done / max(time.time() - start, 1e-9)
                print(f"  {pert.name}: {done}/{len(sample)} images ({rate:.1f}/s)", flush=True)
    if batch:
        flush(batch)
    return out


# ---------------------------------------------------------------- 4. scoring


def score(heads: list[HeadInfo], pert: Perturbation, embeddings: dict, sample: list[dict]) -> pd.DataFrame:
    """Per-image P(fake) of every head under one perturbation; p_fake is NaN where skipped."""
    base = pd.DataFrame({c: [row[c] for row in sample] for c in SAMPLE_COLUMNS})
    frames = []
    for h in heads:
        X = embeddings[h.combo]
        scored = ~np.isnan(X).any(axis=1)
        p_fake = np.full(len(sample), np.nan)
        if scored.any():
            p_fake[scored] = predict_fake(h.head, X[scored])
        frame = base.copy()
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
    local, results = ROOT / cfg3["local_root"], ROOT / cfg3["results_root"]
    local.mkdir(parents=True, exist_ok=True)
    # Keep embeddings and predictions out of Git without editing the shared .gitignore.
    (local / ".gitignore").write_text("*\n", encoding="utf-8")

    perts = parse_perturbations(cfg3["perturbations"])
    names = [p.name for p in perts]
    if args.only and (unknown := set(args.only) - set(names)):
        raise SystemExit(f"--only: unknown perturbations {sorted(unknown)}; choose from {names}")

    heads = load_heads(ROOT, cfg2, cfg3)
    combos = sorted({h.combo for h in heads})
    jpeg = jpeg_by_format(heads)
    per_class = args.per_class if args.per_class is not None else cfg3["sample"]["per_class_per_group"]
    sample = build_sample(ROOT, cfg2, cfg3, per_class)
    ids = [row["image_id"] for row in sample]
    print(f"{len(sample)} test images, {len(heads)} heads, {len(perts)} perturbations, combos {combos}")

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
            embeddings = embed_condition(pert, sample, combos, jpeg, backbone, ROOT, int(cfg3["batch_size"]))
            save_cache(path, signature, ids, embeddings)
            print(f"{pert.name}: embedded in {time.time() - t0:.0f}s")
        elif embeddings is None:
            print(f"{pert.name}: not cached and not in --only; left out of the predictions")
            continue
        else:
            print(f"{pert.name}: reusing cached embeddings")
        frames.append(score(heads, pert, embeddings, sample))
        meta_rows.append({**pert.spec(), "signature": signature, "skipped_per_combo": {
            f"{p}/{f}": int(np.isnan(e).any(axis=1).sum()) for (p, f), e in embeddings.items()}})

    if not frames:
        raise SystemExit("no perturbation has embeddings yet; run without --only first")
    predictions = pd.concat(frames, ignore_index=True)
    out = local / "predictions" / f"{cfg3['backbone']}__{cfg3['protocol']}.csv.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out, index=False)

    # Small, committed records: exactly which images were tested and with what settings.
    write_csv(results / "sample.csv", ({c: row[c] for c in SAMPLE_COLUMNS} for row in sample), SAMPLE_COLUMNS)
    write_json(results / "run_meta.json", {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": args.config, "per_class_per_group": per_class, "seed": cfg3["sample"]["seed"],
        "n_images": len(sample), "heads": [h.name for h in heads],
        "jpeg_recipe": {k: (None if v is None else {"quality": v.quality, "subsampling": v.subsampling})
                        for k, v in jpeg.items()},
        "perturbations": meta_rows,
    })
    print(f"wrote {out.relative_to(ROOT)} ({len(predictions)} rows) and {results.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
