from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from common import project_root_from_script, read_csv


def validate_master(path: Path) -> list[str]:
    rows = read_csv(path)
    errors: list[str] = []
    ids = [row["image_id"] for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("master manifest image_id values are not unique")
    for row in rows:
        if row["label"] not in {"0", "1"}:
            errors.append(f"invalid label for {row['image_id']}: {row['label']}")
        expected = "Real" if row["label"] == "0" else row["generator"]
        if row["label"] == "0" and row["generator"] != "Real":
            errors.append(f"real label has non-real generator: {row['image_id']}")
        if row["label"] == "1" and row["generator"] == "Real":
            errors.append(f"fake label has Real generator: {row['image_id']}")
    return errors


def validate_split(path: Path) -> list[str]:
    rows = read_csv(path)
    errors: list[str] = []
    ids: dict[str, set[str]] = collections.defaultdict(set)
    groups: dict[str, set[str]] = collections.defaultdict(set)
    for row in rows:
        ids[row["split"]].add(row["image_id"])
        groups[row["split"]].add(row["duplicate_group_id"])
    roles = list(ids)
    for index, left in enumerate(roles):
        for right in roles[index + 1:]:
            overlap = ids[left] & ids[right]
            if overlap:
                errors.append(f"{path.name}: {len(overlap)} image IDs overlap {left}/{right}")
            group_overlap = groups[left] & groups[right]
            if group_overlap:
                errors.append(f"{path.name}: {len(group_overlap)} duplicate groups overlap {left}/{right}")
    held = rows[0].get("fold", "") if rows else ""
    if path.name.startswith("logo_"):
        leaked = [row for row in rows if row["generator"].lower() == held.lower() and row["split"] in {"train", "validation"}]
        if leaked:
            errors.append(f"{path.name}: held-out generator leaked into train/validation")
    return errors


def validate_features(index_path: Path, project_root: Path) -> list[str]:
    rows = read_csv(index_path)
    errors: list[str] = []
    by_file: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for row in rows:
        by_file[row["feature_file"]].append(row)
    dimension = None
    for relative, file_rows in by_file.items():
        with np.load(project_root / relative) as archive:
            ids = archive["image_ids"].astype(str)
            embeddings = archive["embeddings"]
        if embeddings.shape[0] != len(ids):
            errors.append(f"{relative}: ID/embedding count mismatch")
        if not np.isfinite(embeddings).all():
            errors.append(f"{relative}: NaN or infinite feature values")
        if dimension is None:
            dimension = embeddings.shape[1]
        elif embeddings.shape[1] != dimension:
            errors.append(f"{relative}: inconsistent feature dimension")
        for row in file_rows:
            index = int(row["feature_row"])
            if index >= len(ids) or ids[index] != row["image_id"]:
                errors.append(f"{relative}: feature ordering mismatch for {row['image_id']}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate manifests, splits, and optional CLIP features.")
    parser.add_argument("--project-root", type=Path, default=project_root_from_script(__file__))
    parser.add_argument("--master-manifest", type=Path)
    parser.add_argument("--feature-index", type=Path, action="append", default=[])
    args = parser.parse_args()
    root = args.project_root.resolve()
    master = args.master_manifest or root / "data" / "manifests" / "master_manifest.csv"
    errors = validate_master(master)
    for split_path in sorted((root / "data" / "splits").glob("*.csv")):
        errors.extend(validate_split(split_path))
    for feature_path in args.feature_index:
        errors.extend(validate_features(feature_path, root))
    report = {"status": "ok" if not errors else "failed", "error_count": len(errors), "errors": errors}
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
