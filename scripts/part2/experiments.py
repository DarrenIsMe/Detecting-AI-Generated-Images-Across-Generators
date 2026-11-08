"""Experiment runner: fit a head on one feature set, score groups on another, reproduce Part 1."""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from common import write_json
from part2.backbones import normalisation
from part2.features import (
    FeatureSet,
    StaleFeatureCache,
    feature_set_id,
    feature_spec,
    load_feature_set_dir,
    spec_hash,
)
from part2.formats import load_jpeg_spec
from part2.head import FittedHead, fit_head, predict_fake, save_head, stratified_subsample
from part2.metrics import group_metrics
from part2.protocols import ProtocolSplit, assert_no_leakage, load_protocol

PRED_COLUMNS = ["image_id", "label", "generator", "evaluation_group", "p_fake", "backbone", "pipeline",
                "format", "test_pipeline", "test_format", "protocol", "fold", "seed"]


def run_id(train_fs: str, test_fs: str, protocol: str, fold: str, seed: int | None) -> str:
    """Stable identifier of one run; cross-condition runs append the test feature set."""
    rid = f"{train_fs}__{protocol}__{fold}__{seed or 'full'}"
    return rid + (f"__x__{test_fs}" if test_fs != train_fs else "")


@dataclass
class RunOutput:
    run_id: str
    head: FittedHead
    metrics: list[dict]
    predictions: pd.DataFrame


def _labels(rows: list[dict]) -> np.ndarray:
    return np.array([int(r["label"]) for r in rows])


def _fs_id(fs: FeatureSet) -> str:
    m = fs.meta
    return feature_set_id(m["backbone"], m["pipeline"], m["format"])


def feature_dir(root: Path, cfg: dict, key: tuple[str, str, str]) -> Path:
    """Cache directory of the feature set ``key`` = (backbone, pipeline, format)."""
    return Path(root) / cfg["local_root"] / "features" / feature_set_id(*key)


def load_feature_meta(root: Path, cfg: dict, key: tuple[str, str, str]) -> dict:
    """A cached feature set's meta.json alone (cheap: no embeddings are read)."""
    path = feature_dir(root, cfg, key) / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: extract {feature_set_id(*key)} with part2_extract.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def load_checked(root: Path, cfg: dict, key: tuple[str, str, str], cache: dict | None = None) -> FeatureSet:
    """Load a cached feature set and verify it was built from the current config and JPEG profile.

    Besides the meta's backbone/pipeline/format and JPEG recipe, the feature spec hash is recomputed
    from the cached ids and the current config, exactly as part2_extract.py computed it. A changed
    backbone or pipeline block (or JPEG profile) therefore raises StaleFeatureCache instead of
    silently training heads, whose metadata copies those blocks, on features made otherwise.
    """
    if cache is not None and key in cache:
        return cache[key]
    backbone, pipeline, fmt = key
    fs = load_feature_set_dir(feature_dir(root, cfg, key))
    got = {k: fs.meta.get(k) for k in ("backbone", "pipeline", "format")}
    if got != {"backbone": backbone, "pipeline": pipeline, "format": fmt}:
        raise StaleFeatureCache(f"{feature_set_id(*key)}: cache meta {got} does not match the request")
    jpeg = None
    if fmt == "matched_jpeg":
        jpeg = load_jpeg_spec(Path(root) / cfg["results_root"] / "jpeg_profile.json")
        if fs.meta.get("jpeg") != dataclasses.asdict(jpeg):
            raise StaleFeatureCache(f"{feature_set_id(*key)}: cached jpeg {fs.meta.get('jpeg')} != profile {jpeg}")
    expected = spec_hash(feature_spec(backbone, pipeline, fmt, cfg, jpeg, [str(i) for i in fs.ids]))
    if fs.meta.get("spec_hash") != expected:
        raise StaleFeatureCache(
            f"{feature_set_id(*key)}: cached spec_hash does not match config/part2.yaml (backbone or "
            "pipeline settings changed since extraction); re-extract with part2_extract.py --overwrite")
    if cache is not None:
        cache[key] = fs
    return fs


def run_fingerprint(train_meta: dict, test_meta: dict, cfg: dict, seed: int | None) -> tuple[dict, str]:
    """Everything a run's outputs depend on besides the code and the committed split CSVs.

    Feature sets are named by id and versioned by spec_hash (settings and image ids) plus
    created_at (so a re-extraction with identical settings still counts as new features); head
    and exclusion settings come from the config. Returns the fields and their short hash. A run
    whose stored fingerprint differs from the current one is stale and is re-run (see run_complete).
    """
    def fs_name(meta: dict) -> str:
        return feature_set_id(meta.get("backbone"), meta.get("pipeline"), meta.get("format"))

    fields = {
        "train_features": fs_name(train_meta),
        "train_spec_hash": train_meta.get("spec_hash"),
        "train_created_at": train_meta.get("created_at"),
        "test_features": fs_name(test_meta),
        "test_spec_hash": test_meta.get("spec_hash"),
        "test_created_at": test_meta.get("created_at"),
        "C_grid": [float(c) for c in cfg["head"]["C_grid"]],
        "max_iter": int(cfg["head"]["max_iter"]),
        "subsample_fraction": float(cfg["subsample_fraction"]),
        "min_side": cfg.get("min_side"),  # None = no exclusion (the reproduction check)
        "seed": "full" if seed is None else int(seed),
    }
    return fields, spec_hash(fields)[:16]


def run_split(train_fs: FeatureSet, test_fs: FeatureSet, split: ProtocolSplit, seed: int | None, cfg: dict) -> RunOutput:
    """Fit on ``train_fs`` (all rows, or a seeded 80% subsample); score test/separate groups on ``test_fs``."""
    if train_fs.meta["backbone"] != test_fs.meta["backbone"]:
        raise ValueError("train and test feature sets come from different backbones: "
                         f"{train_fs.meta['backbone']} vs {test_fs.meta['backbone']}")
    train_rows = split.train if seed is None else stratified_subsample(split.train, cfg["subsample_fraction"], seed)
    ids = lambda rows: [r["image_id"] for r in rows]  # noqa: E731
    head = fit_head(train_fs.rows(ids(train_rows)), _labels(train_rows),
                    train_fs.rows(ids(split.validation)), _labels(split.validation),
                    cfg["head"]["C_grid"], cfg["head"]["max_iter"])

    tm, sm = train_fs.meta, test_fs.meta
    rid = run_id(_fs_id(train_fs), _fs_id(test_fs), split.protocol, split.fold, seed)
    fingerprint, fingerprint_hash = run_fingerprint(tm, sm, cfg, seed)
    seed_label = "full" if seed is None else seed
    backbone_cfg = cfg["backbones"][tm["backbone"]]
    mean, std = normalisation(backbone_cfg["kind"])
    head.meta = {"backbone": tm["backbone"], "pipeline": tm["pipeline"], "format": tm["format"],
                 "protocol": split.protocol, "fold": split.fold, "seed": seed_label,
                 "dim": int(train_fs.X.shape[1]), "train_spec_hash": tm.get("spec_hash"),
                 # Feature-extraction settings (spec section 7); score_images verifies them before
                 # scoring. load_checked guarantees they are the ones the features were built with.
                 "backbone_cfg": dict(backbone_cfg),
                 "pipeline_cfg": dict(cfg["pipelines"][tm["pipeline"]]),
                 "normalisation": {"mean": list(mean), "std": list(std)},
                 "jpeg": tm.get("jpeg"),  # matched-JPEG recipe of the training fakes (None for raw)
                 "fingerprint": fingerprint, "fingerprint_hash": fingerprint_hash}
    # Fields shared by every metric row and prediction of this run.
    run_fields = {"run_id": rid, "backbone": tm["backbone"], "pipeline": tm["pipeline"], "format": tm["format"],
                  "test_pipeline": sm["pipeline"], "test_format": sm["format"], "protocol": split.protocol,
                  "fold": split.fold, "seed": seed_label}

    def metric_row(group: str, y: np.ndarray, p: np.ndarray, on_train: bool = False) -> dict:
        row = {**run_fields, "group": group, "C": head.C, **group_metrics(y, p),
               # which features and settings produced this row (checked by run_complete)
               "fingerprint": fingerprint, "fingerprint_hash": fingerprint_hash}
        if on_train:  # validation is scored on train features: label it with the train condition
            row["test_pipeline"], row["test_format"] = tm["pipeline"], tm["format"]
        return row

    metrics: list[dict] = []
    pred_frames: list[pd.DataFrame] = []
    test_y, test_p = [], []
    # Test groups, then separate groups (which are never part of "overall").
    for kind, groups in (("test", split.test_groups), ("separate", split.separate_groups)):
        for name, rows in groups.items():
            y, p = _labels(rows), predict_fake(head, test_fs.rows(ids(rows)))
            metrics.append(metric_row(name, y, p))
            if kind == "test":
                test_y.append(y)
                test_p.append(p)
            pred_frames.append(pd.DataFrame({
                "image_id": ids(rows), "label": y, "generator": [r["generator"] for r in rows],
                "evaluation_group": name, "p_fake": p}))
    if test_y:
        metrics.append(metric_row("overall", np.concatenate(test_y), np.concatenate(test_p)))
    # Validation is scored on the train feature set, the one the head selected C on.
    metrics.append(metric_row("validation", _labels(split.validation),
                              predict_fake(head, train_fs.rows(ids(split.validation))), on_train=True))

    preds = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame(columns=PRED_COLUMNS[:5])
    for key in PRED_COLUMNS[5:]:
        preds[key] = run_fields[key]
    return RunOutput(rid, head, metrics, preds[PRED_COLUMNS])


def _paths(rid: str, full: bool, root: Path, cfg: dict) -> dict[str, Path]:
    results, local = Path(root) / cfg["results_root"], Path(root) / cfg["local_root"]
    return {"metrics": results / "metrics" / f"{rid}.json",
            "head": (results if full else local) / "heads" / f"{rid}.json",
            "predictions": local / "predictions" / f"{rid}.csv"}


def write_run(out: RunOutput, root: Path, cfg: dict) -> None:
    """Persist a run. Head and predictions go first; the metrics file is the completion marker,
    written last and atomically."""
    full = out.head.meta["seed"] == "full"
    paths = _paths(out.run_id, full, root, cfg)
    save_head(out.head, paths["head"])
    if full:
        paths["predictions"].parent.mkdir(parents=True, exist_ok=True)
        out.predictions.to_csv(paths["predictions"], index=False)
    tmp = paths["metrics"].with_name(paths["metrics"].name + ".tmp")
    write_json(tmp, out.metrics)
    os.replace(tmp, paths["metrics"])


def _stored_fingerprints(paths: dict[str, Path]) -> set:
    """Fingerprint hashes recorded in a run's metric rows and head (None where absent or unreadable)."""
    try:
        rows = json.loads(paths["metrics"].read_text(encoding="utf-8"))
        head_meta = json.loads(paths["head"].read_text(encoding="utf-8")).get("meta", {})
        return {row.get("fingerprint_hash") for row in rows} | {head_meta.get("fingerprint_hash")}
    except (OSError, ValueError, AttributeError, TypeError):  # e.g. a head cut short by an interrupted re-run
        return {None}


def run_complete(rid: str, seed: int | None, split: ProtocolSplit, root: Path, cfg: dict,
                 fingerprint_hash: str) -> bool:
    """True when every output this run should have written exists AND carries ``fingerprint_hash``.

    Outputs written from other features or settings (or before fingerprints existed) are stale:
    the run is reported and must be re-run, so the tables never mix old and new metrics.
    """
    paths = _paths(rid, seed is None, root, cfg)
    needed = [paths["metrics"], paths["head"]]
    if seed is None and (split.test_groups or split.separate_groups):
        needed.append(paths["predictions"])
    if not all(p.exists() for p in needed):
        return False
    if _stored_fingerprints(paths) != {fingerprint_hash}:
        print(f"stale {rid}: re-running", flush=True)
        return False
    return True


def reproduce_part1(cfg: dict, root: Path) -> dict:
    """Part 1 baseline: CLIP B/32, pipeline A, raw, strict, no exclusion, C=1.0. Gate on the stored targets."""
    rep = cfg["reproduction"]
    root = Path(root)
    fs = load_checked(root, cfg, (rep["backbone"], rep["pipeline"], rep["format"]))
    split = load_protocol("strict", root / "data" / "splits")[0]
    assert_no_leakage(split)
    # Fixed C and Part 1's max_iter; min_side None records in the fingerprint that nothing is excluded.
    run_cfg = {**cfg, "head": {"C_grid": [rep["C"]], "max_iter": rep["max_iter"]}, "min_side": None}
    out = run_split(fs, fs, split, None, run_cfg)
    ba = {m["group"]: m["balanced_accuracy"] for m in out.metrics}
    result = {"validation_accuracy": ba["validation"], "test_accuracy": ba["overall"],
              "midjourney_accuracy": ba["Midjourney"]}
    off = {k: (v, rep["targets"][k]) for k, v in result.items() if abs(v - rep["targets"][k]) > rep["tolerance"] + 1e-9}
    if off:
        raise AssertionError(f"Part 1 reproduction failed (got, target): {off}")
    save_head(out.head, root / cfg["results_root"] / "heads" / "reproduction.json")
    return {**result, "targets": rep["targets"], "tolerance": rep["tolerance"], "status": "ok"}
