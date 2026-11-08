r"""Extract and cache frozen-backbone features for Part 2.

    py scripts\part2_extract.py --backbone clip_b32 --pipeline A --format raw
    py scripts\part2_extract.py --all

A cache is reused when its spec hash (config blocks, JPEG recipe, image ids) matches. The hash does
not cover the code, so after changing preprocessing or extraction code re-extract with --overwrite.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

# Silence a harmless Hugging Face symlink warning on Windows (must be set before the imports).
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import dataclasses
import itertools
import json

import numpy as np
import open_clip
import torch
import torchvision

from common import load_config, project_root_from_script
from part2.backbones import load_backbone
from part2.data import iter_image_bytes, load_manifest, small_image_ids
# BACKBONE_ORDER is the --all order (clip_l14 first), shared with part2_run.py.
from part2.features import (
    BACKBONE_ORDER,
    FORMATS,
    PIPELINES,
    extract_features,
    feature_set_id,
    feature_spec,
    save_feature_set,
    spec_hash,
)
from part2.formats import load_jpeg_spec
from part2.preprocess import check_config_matches_code


def expected_ids(rows: list[dict], pipeline: str, min_side: int) -> list[str]:
    """Ids in the order iter_image_bytes yields them, minus too-small images under pipeline B."""
    ordered = sorted(rows, key=lambda r: (r["shard"], int(r["row_in_shard"])))
    drop = small_image_ids(rows, min_side) if pipeline == "B" else set()
    return [r["image_id"] for r in ordered if r["image_id"] not in drop]


def progress_items(rows: list[dict], root, label: str):
    """iter_image_bytes with one progress line per finished shard."""
    per_shard: dict[str, int] = {}
    for row in rows:
        per_shard[row["shard"]] = per_shard.get(row["shard"], 0) + 1
    seen: dict[str, int] = {}
    for row, data in iter_image_bytes(rows, root):
        yield row, data
        seen[row["shard"]] = seen.get(row["shard"], 0) + 1
        if seen[row["shard"]] == per_shard[row["shard"]]:
            print(f"[{label}] shard {row['shard']} done ({sum(seen.values())}/{len(rows)})", flush=True)


def run_one(root, cfg, rows, backbone_name, pipeline, fmt, batch_size, overwrite, backbone_cache) -> None:
    sid = feature_set_id(backbone_name, pipeline, fmt)
    out_dir = root / cfg["local_root"] / "features" / sid
    jpeg = None
    if fmt == "matched_jpeg":
        profile = root / cfg["results_root"] / "jpeg_profile.json"
        if not profile.exists():
            raise SystemExit(f"{profile} not found: run scripts\\part2_jpeg_profile.py first.")
        jpeg = load_jpeg_spec(profile)

    ids_expected = expected_ids(rows, pipeline, cfg["min_side"])
    expected_hash = spec_hash(feature_spec(backbone_name, pipeline, fmt, cfg, jpeg, ids_expected))
    meta_path = out_dir / "meta.json"
    if not overwrite and meta_path.exists():
        cached = json.loads(meta_path.read_text(encoding="utf-8")).get("spec_hash")
        if cached == expected_hash:
            print(f"[{sid}] cache is current; skipping", flush=True)
            return

    # Load each backbone once and reuse it across its sets.
    ext = cfg["extraction"]
    if backbone_name not in backbone_cache:
        backbone_cache.clear()  # free the previous backbone's memory
        backbone_cache[backbone_name] = load_backbone(backbone_name, cfg, root, ext["device"])
    backbone = backbone_cache[backbone_name]

    skip = set() if pipeline == "A" else small_image_ids(rows, cfg["min_side"])
    todo = [r for r in rows if r["image_id"] not in skip]
    print(f"[{sid}] extracting {len(todo)} images", flush=True)
    ids, X, skipped = extract_features(
        progress_items(todo, root, sid), backbone, pipeline, fmt, jpeg,
        batch_size=batch_size or ext["batch_size"], decode_threads=ext["decode_threads"],
    )

    # The ids actually extracted must match the planned list, or the hash would lie.
    if ids != ids_expected:
        raise SystemExit(f"[{sid}] extracted ids differ from the expected list ({len(ids)} vs {len(ids_expected)})")
    meta = {
        "backbone": backbone_name,
        "pipeline": pipeline,
        "format": fmt,
        "jpeg": dataclasses.asdict(jpeg) if jpeg is not None else None,
        "dim": int(X.shape[1]),
        "n": int(X.shape[0]),
        "skipped": skipped,
        "spec_hash": spec_hash(feature_spec(backbone_name, pipeline, fmt, cfg, jpeg, ids)),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "open_clip": getattr(open_clip, "__version__", "unknown"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    save_feature_set(out_dir, ids, X.astype(np.float32), meta)
    print(f"[{sid}] saved n={meta['n']} dim={meta['dim']} skipped={len(skipped)}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", choices=BACKBONE_ORDER)
    parser.add_argument("--pipeline", choices=PIPELINES)
    parser.add_argument("--format", choices=FORMATS, dest="fmt")
    parser.add_argument("--all", action="store_true", help="all 12 sets, clip_l14 first")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true",
                        help="re-extract even if the cache is current (needed after code changes)")
    args = parser.parse_args(argv)
    if args.all:
        combos = list(itertools.product(BACKBONE_ORDER, PIPELINES, FORMATS))
    elif args.backbone and args.pipeline and args.fmt:
        combos = [(args.backbone, args.pipeline, args.fmt)]
    else:
        parser.error("give --backbone, --pipeline and --format, or --all")

    root = project_root_from_script(__file__)
    cfg = load_config(root, "config/part2.yaml")
    check_config_matches_code(cfg)  # fail loudly if config geometry/min_side differ from the code
    rows = load_manifest(root)
    backbone_cache: dict = {}
    for backbone_name, pipeline, fmt in combos:
        run_one(root, cfg, rows, backbone_name, pipeline, fmt, args.batch_size, args.overwrite, backbone_cache)


if __name__ == "__main__":
    main()
