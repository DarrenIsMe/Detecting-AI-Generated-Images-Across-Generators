from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from common import project_root_from_script, read_csv, write_csv, write_json


def load_features(index_path: Path, project_root: Path) -> tuple[dict[str, np.ndarray], dict[str, dict[str, str]]]:
    rows = read_csv(index_path)
    vectors: dict[str, np.ndarray] = {}
    metadata: dict[str, dict[str, str]] = {}
    by_file: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for row in rows:
        if row["image_id"] in metadata:
            raise ValueError(f"Duplicate image_id in feature index: {row['image_id']}")
        by_file[row["feature_file"]].append(row)
        metadata[row["image_id"]] = row
    for relative, file_rows in by_file.items():
        with np.load(project_root / relative) as archive:
            matrix = archive["embeddings"]
            ids = archive["image_ids"].astype(str)
        for row in file_rows:
            index = int(row["feature_row"])
            if index < 0 or index >= len(ids) or index >= len(matrix):
                raise ValueError(f"Feature row out of range for {row['image_id']}: {index}")
            if ids[index] != row["image_id"]:
                raise ValueError(f"Feature mismatch for {row['image_id']}")
            vectors[row["image_id"]] = matrix[index]
    return vectors, metadata


def validate_required_rows(
    rows: list[dict[str, str]],
    vectors: dict[str, np.ndarray],
    manifest_name: str,
    required_splits: tuple[str, ...],
) -> dict[str, dict[int, int]]:
    if not rows:
        raise ValueError(f"{manifest_name}: manifest is empty")
    seen: set[str] = set()
    duplicates: list[str] = []
    for row in rows:
        if row["image_id"] in seen:
            duplicates.append(row["image_id"])
        seen.add(row["image_id"])
    if duplicates:
        raise ValueError(f"{manifest_name}: duplicate image IDs, e.g. {duplicates[:5]}")

    actual_splits = {row["split"] for row in rows}
    expected_splits = set(required_splits)
    if actual_splits != expected_splits:
        raise ValueError(
            f"{manifest_name}: expected splits {sorted(expected_splits)}, "
            f"found {sorted(actual_splits)}"
        )

    missing = [row["image_id"] for row in rows if row["image_id"] not in vectors]
    if missing:
        raise ValueError(
            f"{manifest_name}: feature index is missing {len(missing)} required images; "
            f"examples: {missing[:5]}"
        )

    counts: dict[str, dict[int, int]] = {}
    for split in required_splits:
        split_rows = [row for row in rows if row["split"] == split]
        labels = collections.Counter(int(row["label"]) for row in split_rows)
        invalid_labels = sorted(set(labels) - {0, 1})
        if invalid_labels:
            raise ValueError(f"{manifest_name}: invalid labels in {split}: {invalid_labels}")
        if labels[0] == 0 or labels[1] == 0:
            raise ValueError(
                f"{manifest_name}: {split} requires both classes; "
                f"found real={labels[0]}, fake={labels[1]}"
            )
        counts[split] = {0: labels[0], 1: labels[1]}
    return counts


def dataset_for_split(
    rows: list[dict[str, str]],
    vectors: dict[str, np.ndarray],
    split: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    selected = [row for row in rows if row["split"] == split]
    return (
        np.stack([vectors[row["image_id"]] for row in selected]),
        np.asarray([int(row["label"]) for row in selected]),
        selected,
    )


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed-split linear input-standardization ablation.")
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--feature-index", required=True, type=Path)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--additional-test-manifest", type=Path, action="append", default=[])
    parser.add_argument("--project-root", type=Path, default=project_root_from_script(__file__))
    parser.add_argument("--seeds", nargs="+", type=int, default=[4001, 4002, 4003])
    args = parser.parse_args()
    root = args.project_root.resolve()
    split_rows = read_csv(args.split_manifest)
    vectors, _ = load_features(args.feature_index, root)
    validate_required_rows(
        split_rows, vectors, args.split_manifest.name,
        ("train", "validation", "test"),
    )
    additional_manifests = [
        (path, read_csv(path)) for path in args.additional_test_manifest
    ]
    for path, rows in additional_manifests:
        validate_required_rows(rows, vectors, path.name, ("test",))

    results = []
    for seed in args.seeds:
        datasets = {}
        for role in ("train", "validation", "test"):
            datasets[role] = dataset_for_split(split_rows, vectors, role)
        classifier = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=seed)
        classifier.fit(datasets["train"][0], datasets["train"][1])
        for role in ("validation", "test"):
            predictions = classifier.predict(datasets[role][0])
            result = {"variant": args.variant, "seed": seed, "split": role, **metrics(datasets[role][1], predictions)}
            results.append(result)
            if role == "test":
                by_generator: dict[str, list[int]] = collections.defaultdict(list)
                for index, row in enumerate(datasets[role][2]):
                    by_generator[row.get("evaluation_group") or row["generator"]].append(index)
                for generator, indices in sorted(by_generator.items()):
                    idx = np.asarray(indices)
                    results.append({"variant": args.variant, "seed": seed, "split": "test", "generator": generator, **metrics(datasets[role][1][idx], predictions[idx])})

        for additional_manifest, additional_rows in additional_manifests:
            additional_x = np.stack([vectors[row["image_id"]] for row in additional_rows])
            additional_y = np.asarray([int(row["label"]) for row in additional_rows])
            additional_predictions = classifier.predict(additional_x)
            results.append({
                "variant": args.variant,
                "seed": seed,
                "split": "separate_test",
                "generator": additional_rows[0].get("evaluation_group") or additional_rows[0]["generator"],
                "manifest": additional_manifest.name,
                **metrics(additional_y, additional_predictions),
            })

    output = root / "results" / "input_standardization_ablation" / args.variant
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "metrics.json", results)
    flat = []
    for row in results:
        flat.append({**row, "confusion_matrix": json.dumps(row["confusion_matrix"])})
    write_csv(output / "metrics.csv", flat, sorted({key for row in flat for key in row}))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
