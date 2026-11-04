from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import imagehash
import pyarrow.parquet as pq
from PIL import Image, ImageOps

from common import (
    BKTree,
    GENERATOR_FAMILIES,
    GENERATOR_NAMES,
    UnionFind,
    load_config,
    project_root_from_script,
    resolve_configured_path,
    stable_id,
    write_csv,
    write_json,
)


MANIFEST_FIELDS = [
    "image_id", "source_split", "shard", "row_in_shard", "original_filename",
    "original_uri", "label", "label_name", "generator", "generator_family",
    "source_dataset", "source_image_id", "imagenet_class_id", "width", "height",
    "aspect_ratio", "original_format", "original_mode", "encoded_size_bytes",
    "has_exif", "encoded_sha256", "pixel_sha256", "perceptual_hash",
    "processing_status", "error", "duplicate_group_id", "duplicate_group_kind",
]


def class_identifier(filename: str, generator: str) -> str:
    if generator == "Real":
        match = re.match(r"^(n\d{8})_", filename, re.IGNORECASE)
        return match.group(1).lower() if match else ""
    if generator in {"GLIDE", "VQDM"}:
        match = re.search(r"_(\d{3})_(?:glide|vqdm)_", filename, re.IGNORECASE)
        return match.group(1) if match else ""
    match = re.match(r"^(\d+)_", filename)
    return match.group(1).zfill(3) if match else ""


def pixel_hash(image: Image.Image) -> str:
    payload = f"{image.width}x{image.height}|RGB|".encode("ascii") + image.tobytes()
    return hashlib.sha256(payload).hexdigest()


def build_duplicate_groups(records: list[dict], radius: int) -> list[dict]:
    valid_indices = [index for index, row in enumerate(records) if row["processing_status"] == "ok"]
    exact_union = UnionFind(len(records))
    exact_encoded: dict[str, int] = {}
    exact_pixels: dict[str, int] = {}

    for index in valid_indices:
        row = records[index]
        encoded = row["encoded_sha256"]
        decoded = row["pixel_sha256"]
        if encoded in exact_encoded:
            exact_union.union(index, exact_encoded[encoded])
        else:
            exact_encoded[encoded] = index
        if decoded in exact_pixels:
            exact_union.union(index, exact_pixels[decoded])
        else:
            exact_pixels[decoded] = index

    components: dict[int, list[int]] = collections.defaultdict(list)
    for index in valid_indices:
        components[exact_union.find(index)].append(index)

    duplicate_rows: list[dict] = []
    for members in components.values():
        if len(members) == 1:
            continue
        encoded_count = len({records[index]["encoded_sha256"] for index in members})
        if encoded_count == 1:
            kind = "exact_encoded"
        else:
            kind = "exact_pixels"
        group_key = min(records[index]["image_id"] for index in members)
        group_id = f"{kind}:{group_key}"
        splits = sorted({records[index]["source_split"] for index in members})
        generators = sorted({records[index]["generator"] for index in members})
        labels = sorted({records[index]["label_name"] for index in members})
        for index in members:
            records[index]["duplicate_group_id"] = group_id
            records[index]["duplicate_group_kind"] = kind
        if len(members) > 1:
            duplicate_rows.append({
                "duplicate_group_id": group_id,
                "kind": kind,
                "size": len(members),
                "source_splits": "|".join(splits),
                "generators": "|".join(generators),
                "labels": "|".join(labels),
                "cross_split": len(splits) > 1,
                "cross_generator": len(generators) > 1,
                "cross_label": len(labels) > 1,
                "image_ids": "|".join(records[index]["image_id"] for index in members),
                "filenames": "|".join(records[index]["original_filename"] for index in members),
            })

    # Perceptual hashes are deliberately advisory. They can collide for unrelated,
    # low-detail images, so candidates are reported but never used as split groups.
    candidate_union = UnionFind(len(records))
    candidate_indices: set[int] = set()
    tree = BKTree()
    for index in valid_indices:
        phash_value = int(records[index]["perceptual_hash"], 16)
        for other, _distance in tree.query(phash_value, radius):
            if exact_union.find(index) == exact_union.find(other):
                continue
            candidate_union.union(index, other)
            candidate_indices.update((index, other))
        tree.add(phash_value, index)
    candidate_components: dict[int, list[int]] = collections.defaultdict(list)
    for index in candidate_indices:
        candidate_components[candidate_union.find(index)].append(index)
    for members in candidate_components.values():
        group_key = min(records[index]["image_id"] for index in members)
        splits = sorted({records[index]["source_split"] for index in members})
        generators = sorted({records[index]["generator"] for index in members})
        labels = sorted({records[index]["label_name"] for index in members})
        duplicate_rows.append({
            "duplicate_group_id": f"perceptual_candidate:{group_key}",
            "kind": "perceptual_candidate",
            "size": len(members),
            "source_splits": "|".join(splits),
            "generators": "|".join(generators),
            "labels": "|".join(labels),
            "cross_split": len(splits) > 1,
            "cross_generator": len(generators) > 1,
            "cross_label": len(labels) > 1,
            "image_ids": "|".join(records[index]["image_id"] for index in members),
            "filenames": "|".join(records[index]["original_filename"] for index in members),
        })
    return duplicate_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Tiny-GenImage Parquet shards.")
    parser.add_argument("--project-root", type=Path, default=project_root_from_script(__file__))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--limit", type=int, default=None, help="Audit at most N rows for a smoke test.")
    args = parser.parse_args()

    root = args.project_root.resolve()
    config = load_config(root, args.config)
    dataset = config["dataset"]
    raw_root = resolve_configured_path(
        root, dataset["raw_dir"], "SC4001_TINY_GENIMAGE_DIR"
    )
    parquet_files = sorted(raw_root.glob(dataset["parquet_glob"]))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found below {raw_root}")

    records: list[dict] = []
    count_limit = args.limit
    for shard in parquet_files:
        source_split = "validation" if shard.name.startswith("validation") else "train"
        parquet = pq.ParquetFile(shard)
        row_offset = 0
        for group_index in range(parquet.metadata.num_row_groups):
            table = parquet.read_row_group(group_index, columns=["image", "label", "generator"])
            for local_index, row in enumerate(table.to_pylist()):
                if count_limit is not None and len(records) >= count_limit:
                    break
                row_index = row_offset + local_index
                label = int(row["label"])
                generator_id = int(row["generator"])
                generator = GENERATOR_NAMES.get(generator_id, f"Unknown-{generator_id}")
                label_name = "real" if label == 0 else "fake"
                filename = row["image"]["path"] or f"row-{row_index}"
                data = row["image"]["bytes"]
                image_id = stable_id(dataset["repo_id"], dataset["revision"], source_split, shard.name, row_index, filename)
                record = {
                    "image_id": image_id,
                    "source_split": source_split,
                    "shard": (
                        shard.relative_to(root).as_posix()
                        if shard.is_relative_to(root)
                        else shard.resolve().as_posix()
                    ),
                    "row_in_shard": row_index,
                    "original_filename": filename,
                    "original_uri": f"hf://datasets/{dataset['repo_id']}@{dataset['revision']}/{source_split}/{shard.name}#{row_index}",
                    "label": label,
                    "label_name": label_name,
                    "generator": generator,
                    "generator_family": GENERATOR_FAMILIES.get(generator, "unknown"),
                    "source_dataset": "Tiny-GenImage",
                    "source_image_id": Path(filename).stem,
                    "imagenet_class_id": class_identifier(filename, generator),
                    "width": "", "height": "", "aspect_ratio": "",
                    "original_format": "", "original_mode": "", "encoded_size_bytes": len(data or b""),
                    "has_exif": False, "encoded_sha256": "", "pixel_sha256": "",
                    "perceptual_hash": "", "processing_status": "ok", "error": "",
                    "duplicate_group_id": f"unique:{image_id}", "duplicate_group_kind": "unique",
                }
                try:
                    if not data:
                        raise ValueError("Image byte field is empty")
                    record["encoded_sha256"] = hashlib.sha256(data).hexdigest()
                    with Image.open(io.BytesIO(data)) as probe:
                        record["original_format"] = probe.format or "unknown"
                        record["original_mode"] = probe.mode
                        probe.verify()
                    with Image.open(io.BytesIO(data)) as decoded:
                        record["has_exif"] = bool(decoded.getexif())
                        decoded = ImageOps.exif_transpose(decoded).convert("RGB")
                        record["width"], record["height"] = decoded.size
                        record["aspect_ratio"] = round(decoded.width / decoded.height, 8)
                        record["pixel_sha256"] = pixel_hash(decoded)
                        record["perceptual_hash"] = str(imagehash.phash(decoded, hash_size=int(config["audit"]["perceptual_hash_size"])))
                    expected_label = 0 if generator == "Real" else 1
                    if label != expected_label:
                        raise ValueError(f"Inconsistent label={label} for generator={generator}")
                except Exception as exc:  # keep successful rows when one image fails
                    record["processing_status"] = "error"
                    record["error"] = f"{type(exc).__name__}: {exc}"
                records.append(record)
            row_offset += table.num_rows
            if count_limit is not None and len(records) >= count_limit:
                break
        print(f"audited {shard.name}: total records={len(records)}", flush=True)
        if count_limit is not None and len(records) >= count_limit:
            break

    duplicate_rows = build_duplicate_groups(records, int(config["audit"]["near_duplicate_hamming_distance"]))
    manifest_dir = root / "data" / "manifests"
    result_dir = root / "results" / "dataset_statistics"
    suffix = "_smoke" if args.limit else ""
    manifest_path = manifest_dir / f"master_manifest{suffix}.csv"
    write_csv(manifest_path, records, MANIFEST_FIELDS)

    duplicate_fields = [
        "duplicate_group_id", "kind", "size", "source_splits", "generators", "labels",
        "cross_split", "cross_generator", "cross_label", "image_ids", "filenames",
    ]
    write_csv(result_dir / f"duplicate_groups{suffix}.csv", duplicate_rows, duplicate_fields)
    errors = [row for row in records if row["processing_status"] != "ok"]
    write_csv(result_dir / f"corrupt_or_unreadable{suffix}.csv", errors, MANIFEST_FIELDS)

    generator_counts = collections.Counter((row["source_split"], row["label_name"], row["generator"]) for row in records)
    format_counts = collections.Counter((row["label_name"], row["generator"], row["original_format"]) for row in records if row["processing_status"] == "ok")
    dimensions = collections.Counter((row["generator"], row["width"], row["height"]) for row in records if row["processing_status"] == "ok")
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset["name"],
        "repo_id": dataset["repo_id"],
        "revision": dataset["revision"],
        "license": dataset["license"],
        "parquet_files": len(parquet_files),
        "rows_audited": len(records),
        "valid_images": len(records) - len(errors),
        "corrupt_or_unreadable": len(errors),
        "unique_original_filenames": len({row["original_filename"].lower() for row in records}),
        "unique_encoded_hashes": len({row["encoded_sha256"] for row in records if row["encoded_sha256"]}),
        "unique_pixel_hashes": len({row["pixel_sha256"] for row in records if row["pixel_sha256"]}),
        "duplicate_groups": len(duplicate_rows),
        "exact_duplicate_groups": sum(row["kind"].startswith("exact_") for row in duplicate_rows),
        "perceptual_candidate_groups": sum(row["kind"] == "perceptual_candidate" for row in duplicate_rows),
        "cross_split_duplicate_groups": sum(str(row["cross_split"]).lower() == "true" or row["cross_split"] is True for row in duplicate_rows),
        "cross_generator_duplicate_groups": sum(str(row["cross_generator"]).lower() == "true" or row["cross_generator"] is True for row in duplicate_rows),
        "cross_label_duplicate_groups": sum(str(row["cross_label"]).lower() == "true" or row["cross_label"] is True for row in duplicate_rows),
        "counts": [
            {"source_split": key[0], "label": key[1], "generator": key[2], "count": value}
            for key, value in sorted(generator_counts.items())
        ],
        "format_counts": [
            {"label": key[0], "generator": key[1], "format": key[2], "count": value}
            for key, value in sorted(format_counts.items())
        ],
        "top_dimensions": [
            {"generator": key[0], "width": key[1], "height": key[2], "count": value}
            for key, value in dimensions.most_common(100)
        ],
        "known_limitations": [
            "Tiny-GenImage does not retain an explicit fake-to-real source pairing.",
            "ImageNet class identifiers are names for real images but numeric indices for generated images; no local mapping joins them.",
            "Perceptual-hash groups are candidates and may contain false positives; review cross-split groups manually.",
            "Midjourney is treated as proprietary/unknown architecture, not as confirmed diffusion.",
            "SD14 exists in the label vocabulary but has zero rows.",
        ],
    }
    write_json(result_dir / f"audit_summary{suffix}.json", summary)

    lines = [
        "# Tiny-GenImage audit summary", "",
        f"- Revision: `{dataset['revision']}`",
        f"- Rows audited: {len(records):,}",
        f"- Readable images: {len(records) - len(errors):,}",
        f"- Corrupt/unreadable: {len(errors):,}",
        f"- Duplicate candidate groups: {len(duplicate_rows):,}",
        f"- Exact duplicate groups: {summary['exact_duplicate_groups']:,}",
        f"- Perceptual candidates requiring review: {summary['perceptual_candidate_groups']:,}",
        f"- Cross-split duplicate groups: {summary['cross_split_duplicate_groups']:,}",
        f"- Cross-generator duplicate groups: {summary['cross_generator_duplicate_groups']:,}",
        f"- Cross-label duplicate groups: {summary['cross_label_duplicate_groups']:,}",
        "", "## Counts", "",
        "| Source split | Label | Generator | Count |", "|---|---|---|---:|",
    ]
    lines.extend(f"| {item['source_split']} | {item['label']} | {item['generator']} | {item['count']:,} |" for item in summary["counts"])
    lines.extend(["", "## Known limitations", ""])
    lines.extend(f"- {item}" for item in summary["known_limitations"])
    (result_dir / f"AUDIT_SUMMARY{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
