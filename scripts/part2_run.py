"""CLI for Part 2 experiments: reproduction gate, single runs, and the full grid."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common import load_config, project_root_from_script, write_json
from part2.data import load_manifest, small_image_ids
from part2.experiments import (
    load_checked,
    load_feature_meta,
    reproduce_part1,
    run_complete,
    run_fingerprint,
    run_id,
    run_split,
    write_run,
)
# Grid order (clip_l14 first: the single-image interface needs it), shared with part2_extract.py.
from part2.features import BACKBONE_ORDER, FORMATS, PIPELINES, feature_set_id
from part2.preprocess import check_config_matches_code
from part2.protocols import assert_no_leakage, load_protocol


def grid_jobs(cfg: dict, only_backbone: str | None = None) -> list[dict]:
    """Ordered job list. A job = train/test feature-set keys (backbone, pipeline, format), protocol, with_seeds."""
    jobs: list[dict] = []
    backbones = [b for b in BACKBONE_ORDER if b in cfg["backbones"] and only_backbone in (None, b)]
    for b in backbones:
        for p in PIPELINES:
            for f in FORMATS:
                for protocol in ("strict", "logo"):
                    jobs.append({"train": (b, p, f), "test": (b, p, f), "protocol": protocol, "with_seeds": True})
        # Cross-condition pairs: full-data only.
        for tr, te in ((("A", "raw"), ("A", "matched_jpeg")), (("B", "raw"), ("B", "matched_jpeg")),
                       (("A", "raw"), ("B", "raw")), (("B", "raw"), ("A", "raw"))):
            for protocol in ("strict", "logo"):
                jobs.append({"train": (b, *tr), "test": (b, *te), "protocol": protocol, "with_seeds": False})
        if b == "clip_l14":  # the final (all-generator) heads belong to the clip_l14 block
            for p in PIPELINES:
                for f in FORMATS:
                    jobs.append({"train": (b, p, f), "test": (b, p, f), "protocol": "final", "with_seeds": False})
    return jobs


def excluded_counts(full_splits: list, exclude: frozenset[str]) -> dict:
    """Per fold, how many rows of each list (train, validation, each group) are excluded ids."""
    def lists(s):
        return {"train": s.train, "validation": s.validation,
                **{f"test:{g}": r for g, r in s.test_groups.items()},
                **{f"separate:{g}": r for g, r in s.separate_groups.items()}}
    return {s.fold: {k: sum(r["image_id"] in exclude for r in rows) for k, rows in lists(s).items()}
            for s in full_splits}


def execute(root: Path, cfg: dict, jobs: list[dict], seeds: list[int], overwrite: bool) -> None:
    """Run jobs in order. Unless overwrite, a run is skipped only when its outputs exist and carry
    the current run fingerprint (same feature extractions, head and exclusion settings); outputs
    from older features or settings are reported as stale and re-run."""
    exclude = frozenset(small_image_ids(load_manifest(root), cfg["min_side"]))
    splits_dir = root / "data" / "splits"
    exc_path = root / cfg["results_root"] / "exclusions.json"
    exclusions = json.loads(exc_path.read_text(encoding="utf-8")) if exc_path.exists() else {}
    protocols: dict[str, list] = {}
    cache: dict = {}
    metas: dict = {}  # feature-set meta.json, read once: enough to fingerprint without loading features
    for job in jobs:
        protocol = job["protocol"]
        if protocol not in protocols:
            protocols[protocol] = load_protocol(protocol, splits_dir, exclude)
            for split in protocols[protocol]:
                assert_no_leakage(split)
            exclusions[protocol] = excluded_counts(load_protocol(protocol, splits_dir), exclude)
            write_json(exc_path, exclusions)
        for key in (job["train"], job["test"]):
            if key not in metas:
                metas[key] = load_feature_meta(root, cfg, key)
        tr_id, te_id = feature_set_id(*job["train"]), feature_set_id(*job["test"])
        for split in protocols[protocol]:
            for seed in [None, *(seeds if job["with_seeds"] else [])]:
                rid = run_id(tr_id, te_id, protocol, split.fold, seed)
                _, fingerprint_hash = run_fingerprint(metas[job["train"]], metas[job["test"]], cfg, seed)
                if not overwrite and run_complete(rid, seed, split, root, cfg, fingerprint_hash):
                    print(f"skip {rid}")
                    continue
                out = run_split(load_checked(root, cfg, job["train"], cache),
                                load_checked(root, cfg, job["test"], cache), split, seed, cfg)
                write_run(out, root, cfg)
                print(f"done {rid}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reproduce", action="store_true", help="Part 1 reproduction gate")
    ap.add_argument("--backbone")
    ap.add_argument("--pipeline", choices=PIPELINES)
    ap.add_argument("--format", choices=FORMATS)
    ap.add_argument("--protocol", choices=["strict", "logo", "final"])
    ap.add_argument("--test-pipeline", choices=PIPELINES)
    ap.add_argument("--test-format", choices=FORMATS)
    ap.add_argument("--no-seeds", action="store_true", help="full-data fit only")
    ap.add_argument("--grid", action="store_true", help="run every configured experiment")
    ap.add_argument("--only-backbone", choices=BACKBONE_ORDER)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    root = project_root_from_script(__file__)
    cfg = load_config(root, "config/part2.yaml")
    check_config_matches_code(cfg)  # fail loudly if config geometry/min_side differ from the code
    if args.reproduce:
        result = reproduce_part1(cfg, root)
        write_json(root / cfg["results_root"] / "reproduction.json", result)
        print(json.dumps(result, indent=2))
        return 0
    if args.grid:
        jobs = grid_jobs(cfg, args.only_backbone)
    else:
        if not (args.backbone and args.pipeline and args.format and args.protocol):
            ap.error("give --reproduce, --grid, or --backbone/--pipeline/--format/--protocol")
        train = (args.backbone, args.pipeline, args.format)
        test = (args.backbone, args.test_pipeline or args.pipeline, args.test_format or args.format)
        jobs = [{"train": train, "test": test, "protocol": args.protocol, "with_seeds": not args.no_seeds}]
    execute(root, cfg, jobs, list(cfg["seeds"]), args.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
