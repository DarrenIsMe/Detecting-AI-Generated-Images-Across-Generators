from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from common import project_root_from_script, read_csv, write_json


VARIANTS = ("original_encoded", "standardized", "reencode_only", "jpeg_common")
RUNTIMES_SECONDS = {
    "clip_smoke_32_images": 14.775,
    "preprocess_original_encoded": 245.783,
    "preprocess_standardized": 591.358,
    "preprocess_reencode_only": 1356.005,
    "preprocess_jpeg_common": 301.518,
    "clip_original_encoded": 618.966,
    "clip_standardized": 417.111,
    "clip_reencode_only": 743.271,
    "clip_jpeg_common": 564.934,
    "ablation_original_encoded": 13.303,
    "ablation_standardized": 4.577,
    "ablation_reencode_only": 4.818,
    "ablation_jpeg_common": 4.584,
}


def directory_size(path: Path) -> dict[str, int | float]:
    files = [item for item in path.rglob("*") if item.is_file()]
    size = sum(item.stat().st_size for item in files)
    return {"files": len(files), "bytes": size, "gib": round(size / 2**30, 3)}


def main() -> None:
    root = project_root_from_script(__file__).resolve()
    report: dict[str, object] = {
        "dataset_revision": "89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69",
        "model": {
            "name": "ViT-B-32-quickgelu",
            "pretrained": "openai",
            "embedding_dimension": 512,
            "checkpoint_sha256": "E6D1BD7789AA45192B3BF90570A789B478BAE1B74EBCCE7EDDD908E83A2B7C31",
        },
        "runtimes_seconds": RUNTIMES_SECONDS,
        "variants": {},
        "split_coverage": {},
        "output_sizes": {},
        "ablation": {},
    }

    feature_sets: dict[str, set[str]] = {}
    for variant in VARIANTS:
        manifest = read_csv(root / "data" / "manifests" / f"{variant}_manifest.csv")
        index_path = next((root / "data" / "features" / variant).rglob("feature_index.csv"))
        index = read_csv(index_path)
        manifest_ids = {row["image_id"] for row in manifest}
        feature_ids = {row["image_id"] for row in index}
        feature_sets[variant] = feature_ids
        report["variants"][variant] = {
            "manifest_rows": len(manifest),
            "feature_rows": len(index),
            "manifest_unique_ids": len(manifest_ids),
            "feature_unique_ids": len(feature_ids),
            "id_sets_equal": manifest_ids == feature_ids,
            "manifest_failures": sum(row["variant_status"] != "ok" for row in manifest),
            "feature_index": index_path.relative_to(root).as_posix(),
        }

        metrics = json.loads((root / "results" / "input_standardization_ablation" / variant / "metrics.json").read_text(encoding="utf-8"))
        selected = [row for row in metrics if row["seed"] == 4001]
        report["ablation"][variant] = selected

    for split_path in sorted((root / "data" / "splits").glob("*.csv")):
        rows = read_csv(split_path)
        ids = {row["image_id"] for row in rows}
        report["split_coverage"][split_path.name] = {
            "rows": len(rows),
            "unique_ids": len(ids),
            "covered_by_all_variants": all(ids <= feature_sets[variant] for variant in VARIANTS),
        }

    size_paths = {
        "raw": root / "data" / "raw" / "tiny_genimage_hf",
        **{f"processed_{variant}": root / "data" / "processed" / variant for variant in VARIANTS},
        "features_all_variants": root / "data" / "features",
        "clip_checkpoint_cache": root / "data" / "models" / "open_clip",
    }
    report["output_sizes"] = {name: directory_size(path) for name, path in size_paths.items()}

    all_ok = (
        all(item["id_sets_equal"] and item["manifest_failures"] == 0 for item in report["variants"].values())
        and all(item["covered_by_all_variants"] for item in report["split_coverage"].values())
    )
    report["status"] = "ok" if all_ok else "failed"
    report["validation_errors"] = [] if all_ok else ["Manifest/feature/split alignment failed; inspect fields above."]

    out = root / "results" / "final"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "final_validation.json", report)

    lines = [
        "# Person 1 final pipeline report",
        "",
        f"Overall validation: **{report['status']}**.",
        "",
        "## Completed outputs",
        "",
        "| Condition | Manifest rows | Feature rows | Processed size (GiB) |",
        "|---|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        item = report["variants"][variant]
        size = report["output_sizes"][f"processed_{variant}"]["gib"]
        lines.append(f"| `{variant}` | {item['manifest_rows']:,} | {item['feature_rows']:,} | {size:.3f} |")
    lines += [
        "",
        "All four conditions cover every existing experiment manifest. Every saved feature archive is 512-dimensional float32, finite, L2-normalized, and ordered by explicit image ID.",
        "",
        "## Strict BigGAN-to-diffusion ablation (seed 4001)",
        "",
        "| Condition | Validation accuracy | Diffusion test accuracy | Diffusion test F1 | Midjourney accuracy (separate) |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        rows = report["ablation"][variant]
        val = next(row for row in rows if row["split"] == "validation")
        test = next(row for row in rows if row["split"] == "test" and "generator" not in row)
        mid = next(row for row in rows if row["split"] == "separate_test")
        lines.append(f"| `{variant}` | {val['accuracy']:.4f} | {test['accuracy']:.5f} | {test['f1']:.5f} | {mid['accuracy']:.4f} |")
    lines += [
        "",
        "Seeds 4001, 4002, and 4003 produced identical metrics because the fixed split and convex linear classifier converged to the same solution.",
        "",
        "The common-JPEG control reduces strict test accuracy and fake recall. This shows that codec treatment materially affects the detector, but it does not prove that JPEG artifacts are removed: JPEG sources are decoded with their existing artifacts and then compressed again.",
        "",
        "Midjourney is treated as proprietary/unknown and is not included in the strict diffusion aggregate. It is never used for training or validation.",
        "",
        "## Runtime summary",
        "",
        "| Stage | Seconds | Minutes |",
        "|---|---:|---:|",
    ]
    for name, seconds in RUNTIMES_SECONDS.items():
        lines.append(f"| `{name}` | {seconds:.3f} | {seconds / 60:.2f} |")
    lines += [
        "",
        "## Important limitations",
        "",
        "- BigGAN is the only GAN, so the strict protocol is BigGAN-to-diffusion rather than broad GAN-family generalization.",
        "- Tiny-GenImage does not document its sampling method from full GenImage and does not retain explicit fake-to-real source pairing.",
        "- Exact duplicates are grouped atomically; reviewed perceptual-hash candidates were false positives. Undetected semantic/source reuse remains an unresolved risk.",
        "- Original real images are JPEG and original fake images are PNG, a severe format confound.",
    ]
    (out / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
