from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, ImageOps

from common import load_config, project_root_from_script, read_csv, write_csv


VARIANTS = {"original_encoded", "reencode_only", "standardized", "jpeg_common"}


def standardized_image(image: Image.Image, size: int) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    scale = size / min(image.size)
    resized = image.resize(
        (max(size, round(image.width * scale)), max(size, round(image.height * scale))),
        Image.Resampling.BICUBIC,
    )
    left = (resized.width - size) // 2
    top = (resized.height - size) // 2
    return resized.crop((left, top, left + size, top + size))


def extension_for_format(format_name: str, fallback: str = ".img") -> str:
    return {
        "JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp",
        "TIFF": ".tif",
    }.get(format_name.upper(), fallback)


def process_bytes(
    data: bytes,
    variant: str,
    size: int,
    compress_level: int,
    jpeg_quality: int,
) -> tuple[bytes, str, int, int]:
    if variant == "original_encoded":
        with Image.open(io.BytesIO(data)) as image:
            format_name = image.format or "unknown"
            width, height = image.size
        return data, extension_for_format(format_name), width, height

    with Image.open(io.BytesIO(data)) as image:
        if variant in {"reencode_only", "jpeg_common"}:
            output = ImageOps.exif_transpose(image).convert("RGB")
        elif variant == "standardized":
            output = standardized_image(image, size)
        else:
            raise ValueError(f"Unsupported variant: {variant}")
        buffer = io.BytesIO()
        if variant == "jpeg_common":
            output.save(
                buffer,
                format="JPEG",
                quality=jpeg_quality,
                subsampling=0,
                optimize=False,
                progressive=False,
            )
            extension = ".jpg"
        else:
            output.save(buffer, format="PNG", compress_level=compress_level)
            extension = ".png"
        return buffer.getvalue(), extension, output.width, output.height


def main() -> None:
    parser = argparse.ArgumentParser(description="Create input-standardization variants without modifying raw data.")
    parser.add_argument("variant", choices=sorted(VARIANTS))
    parser.add_argument("--project-root", type=Path, default=project_root_from_script(__file__))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    config = load_config(root, args.config)
    manifest_path = args.manifest or root / "data" / "manifests" / "master_manifest.csv"
    rows = [row for row in read_csv(manifest_path) if row["processing_status"] == "ok"]
    if args.limit:
        rows = rows[: args.limit]

    by_shard: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_shard.setdefault(row["shard"], []).append(row)

    output_root = root / "data" / "processed" / args.variant
    variant_rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    for shard_name, selected in sorted(by_shard.items()):
        shard_path = root / shard_name
        table = pq.read_table(shard_path, columns=["image"])
        images = table.column("image").to_pylist()
        for row in selected:
            row_index = int(row["row_in_shard"])
            source = images[row_index]
            try:
                if args.dry_run:
                    output_path = Path("<dry-run>") / row["image_id"]
                    width, height = int(row["width"]), int(row["height"])
                else:
                    payload, extension, width, height = process_bytes(
                        source["bytes"], args.variant,
                        int(config["processing"]["output_size"]),
                        int(config["processing"]["png_compress_level"]),
                        int(config["processing"].get("jpeg_quality", 95)),
                    )
                    output_path = output_root / row["generator"] / f"{row['image_id']}{extension}"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    if args.overwrite or not output_path.exists():
                        output_path.write_bytes(payload)
                variant_rows.append({
                    **row,
                    "variant": args.variant,
                    "image_path": output_path.as_posix() if args.dry_run else output_path.relative_to(root).as_posix(),
                    "variant_width": width,
                    "variant_height": height,
                    "variant_status": "ok",
                    "variant_error": "",
                })
            except Exception as exc:
                failed = {**row, "variant": args.variant, "image_path": "", "variant_width": "", "variant_height": "", "variant_status": "error", "variant_error": f"{type(exc).__name__}: {exc}"}
                variant_rows.append(failed)
                failures.append(failed)
        print(f"processed {shard_name}: selected={len(selected)}", flush=True)

    if not args.dry_run:
        fields = list(variant_rows[0].keys()) if variant_rows else []
        write_csv(root / "data" / "manifests" / f"{args.variant}_manifest.csv", variant_rows, fields)
        if failures:
            write_csv(root / "results" / "dataset_statistics" / f"{args.variant}_failures.csv", failures, fields)
    print(f"variant={args.variant} rows={len(variant_rows)} failures={len(failures)} dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
