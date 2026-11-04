from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from common import (
    load_config,
    project_root_from_script,
    read_csv,
    resolve_configured_path,
    write_csv,
    write_json,
)


CACHE_SCHEMA_VERSION = 2
LEGACY_CHECKPOINT_SHA256 = "E6D1BD7789AA45192B3BF90570A789B478BAE1B74EBCCE7EDDD908E83A2B7C31"
LEGACY_PREPROCESS_SHA256 = "c4c7c2276f27ec1b7f9e45f1f76dde5176ca32a245eafeadd607a3c7f4c1660e"
LEGACY_PROCESSING = {
    "original_encoded": {"variant": "original_encoded"},
    "standardized": {
        "variant": "standardized", "output_size": 224,
        "resize_method": "bicubic", "crop_method": "center",
        "png_compress_level": 6,
    },
    "reencode_only": {
        "variant": "reencode_only", "png_compress_level": 6,
    },
    "jpeg_common": {
        "variant": "jpeg_common", "jpeg_quality": 95,
    },
}


def require_clip_dependencies():
    try:
        import torch
        import open_clip
    except ImportError as exc:
        raise SystemExit(
            "CLIP dependencies are not installed. After approval, install torch, torchvision, "
            "and open_clip_torch. No weights are downloaded by this script until it is run."
        ) from exc
    return torch, open_clip


def config_hash(model_name: str, pretrained: str, variant: str, normalize: bool) -> str:
    payload = json.dumps({
        "model_name": model_name, "pretrained": pretrained,
        "variant": variant, "l2_normalize": normalize,
    }, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_fingerprint(rows: list[dict[str, str]], root: Path) -> str:
    """Bind a cache to ordered image IDs and the exact processed bytes on disk."""
    digest = hashlib.sha256()
    for row in rows:
        path = root / row["image_path"]
        digest.update(row["image_id"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def processing_spec(config: dict[str, Any], variant: str) -> dict[str, Any]:
    processing = config["processing"]
    if variant == "original_encoded":
        return {"variant": variant}
    if variant == "standardized":
        return {
            "variant": variant,
            "output_size": int(processing["output_size"]),
            "resize_method": processing["resize_method"],
            "crop_method": processing["crop_method"],
            "png_compress_level": int(processing["png_compress_level"]),
        }
    if variant == "reencode_only":
        return {
            "variant": variant,
            "png_compress_level": int(processing["png_compress_level"]),
        }
    if variant == "jpeg_common":
        return {
            "variant": variant,
            "jpeg_quality": int(processing["jpeg_quality"]),
        }
    raise ValueError(f"Unknown processing variant: {variant}")


def checkpoint_spec(open_clip: Any, cache_dir: Path, model_name: str, pretrained: str) -> dict[str, Any]:
    pretrained_config = open_clip.get_pretrained_cfg(model_name, pretrained) or {}
    search_root = cache_dir
    repository = pretrained_config.get("hf_hub")
    if repository:
        candidate_root = cache_dir / f"models--{repository.replace('/', '--')}"
        if candidate_root.exists():
            search_root = candidate_root
    candidates = sorted({
        path.resolve()
        for suffix in ("*.safetensors", "*.pt", "*.pth", "*.bin")
        for path in search_root.rglob(suffix)
        if path.is_file()
    })
    if not candidates:
        raise RuntimeError(
            f"Could not locate the downloaded checkpoint below {search_root}; "
            "cache safety requires hashing the actual weight file."
        )
    return {
        "pretrained_config_hash": canonical_hash(pretrained_config),
        "files": [
            {"name": path.name, "size": path.stat().st_size, "sha256": file_sha256(path)}
            for path in candidates
        ],
    }


def load_image_batch(
    rows: list[dict[str, str]],
    start: int,
    batch_size: int,
    root: Path,
    preprocess: Callable[[Image.Image], Any],
) -> tuple[int, list[Any], list[dict[str, str]], list[dict[str, str]]]:
    """Load one attempted batch and always return its next input position."""
    batch_rows = rows[start:start + batch_size]
    tensors: list[Any] = []
    usable_rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    for row in batch_rows:
        try:
            path = root / row["image_path"]
            with Image.open(path) as image:
                tensors.append(preprocess(image.convert("RGB")))
            usable_rows.append(row)
        except Exception as exc:
            failures.append({"image_id": row["image_id"], "error": f"{type(exc).__name__}: {exc}"})
    return start + len(batch_rows), tensors, usable_rows, failures


def arrays_are_valid(
    metadata: dict[str, Any],
    cached_ids: list[str],
    cached_embeddings: np.ndarray,
    expected_ids: list[str],
    expected_dimension: int,
    cfg_hash: str,
) -> bool:
    return (
        metadata.get("config_hash") == cfg_hash
        and metadata.get("image_ids") == expected_ids
        and cached_ids == expected_ids
        and cached_embeddings.ndim == 2
        and cached_embeddings.shape == (len(expected_ids), expected_dimension)
        and np.isfinite(cached_embeddings).all()
    )


def legacy_attestation(root: Path, variant: str, shard_name: str) -> dict[str, str] | None:
    path = root / "results" / "clip_model" / "verified_legacy_cache.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("variants", {}).get(variant, {}).get(shard_name)


def can_upgrade_verified_legacy_cache(
    metadata: dict[str, Any],
    variant: str,
    processing: dict[str, Any],
    preprocess_sha256: str,
    checkpoint: dict[str, Any],
    input_sha256: str,
    feature_sha256: str,
    attestation: dict[str, str] | None,
) -> bool:
    checkpoint_hashes = {item["sha256"].upper() for item in checkpoint.get("files", [])}
    return bool(
        attestation
        and metadata.get("model_name") == "ViT-B-32-quickgelu"
        and metadata.get("pretrained") == "openai"
        and metadata.get("variant") == variant
        and bool(metadata.get("l2_normalize"))
        and processing == LEGACY_PROCESSING[variant]
        and preprocess_sha256 == LEGACY_PREPROCESS_SHA256
        and LEGACY_CHECKPOINT_SHA256 in checkpoint_hashes
        and attestation.get("input_fingerprint") == input_sha256
        and attestation.get("feature_file_sha256") == feature_sha256
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen OpenCLIP image embeddings, one source shard at a time.")
    parser.add_argument("variant", choices=["original_encoded", "reencode_only", "standardized", "jpeg_common"])
    parser.add_argument("--project-root", type=Path, default=project_root_from_script(__file__))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    torch, open_clip = require_clip_dependencies()
    root = args.project_root.resolve()
    config = load_config(root, args.config)
    clip_config = config["clip"]
    model_name = clip_config["model_name"]
    pretrained = clip_config["pretrained"]
    normalize = bool(clip_config.get("l2_normalize", True))
    batch_size = args.batch_size or int(clip_config["batch_size"])
    cache_dir = resolve_configured_path(
        root,
        clip_config.get("cache_dir", "data/models/open_clip"),
        "SC4001_CLIP_CACHE_DIR",
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or clip_config.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    manifest_path = args.manifest or root / "data" / "manifests" / f"{args.variant}_manifest.csv"
    rows = [row for row in read_csv(manifest_path) if row.get("variant_status", "ok") == "ok"]
    if args.limit:
        rows = rows[: args.limit]

    # Keep the original directory hash stable so verified caches remain discoverable.
    cfg_hash = config_hash(model_name, pretrained, args.variant, normalize)
    output_root = root / "data" / "features" / args.variant / f"{model_name}-{pretrained}-{cfg_hash}"
    output_root.mkdir(parents=True, exist_ok=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device, cache_dir=str(cache_dir)
    )
    model.eval()
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    expected_dimension = int(model.visual.output_dim)
    preprocess_sha256 = hashlib.sha256(repr(preprocess).encode("utf-8")).hexdigest()
    checkpoint = checkpoint_spec(open_clip, cache_dir, model_name, pretrained)
    processing = processing_spec(config, args.variant)

    by_shard: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_shard.setdefault(Path(row["shard"]).stem, []).append(row)

    index_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for shard_name, shard_rows in sorted(by_shard.items()):
        feature_file = output_root / f"{shard_name}.npz"
        metadata_file = output_root / f"{shard_name}.json"
        expected_ids = [row["image_id"] for row in shard_rows]
        inputs_sha256 = input_fingerprint(shard_rows, root)
        if feature_file.exists() and metadata_file.exists():
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            try:
                feature_sha256 = file_sha256(feature_file)
                with np.load(feature_file) as archive:
                    cached_ids = archive["image_ids"].astype(str).tolist()
                    cached_embeddings = archive["embeddings"]
                    arrays_valid = arrays_are_valid(
                        metadata, cached_ids, cached_embeddings, expected_ids,
                        expected_dimension, cfg_hash,
                    )
                cache_spec = {
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "model_name": model_name,
                    "pretrained": pretrained,
                    "checkpoint": checkpoint,
                    "preprocess_sha256": preprocess_sha256,
                    "processing": processing,
                    "variant": args.variant,
                    "l2_normalize": normalize,
                    "embedding_dimension": expected_dimension,
                    "input_fingerprint": inputs_sha256,
                }
                expected_spec_hash = canonical_hash(cache_spec)
                cache_valid = bool(
                    arrays_valid
                    and metadata.get("cache_schema_version") == CACHE_SCHEMA_VERSION
                    and metadata.get("cache_spec_hash") == expected_spec_hash
                    and metadata.get("input_fingerprint") == inputs_sha256
                    and metadata.get("feature_file_sha256") == feature_sha256
                )
                if not cache_valid and arrays_valid and can_upgrade_verified_legacy_cache(
                    metadata, args.variant, processing, preprocess_sha256, checkpoint,
                    inputs_sha256, feature_sha256,
                    legacy_attestation(root, args.variant, shard_name),
                ):
                    metadata.update({
                        "cache_schema_version": CACHE_SCHEMA_VERSION,
                        "cache_spec_hash": expected_spec_hash,
                        "input_fingerprint": inputs_sha256,
                        "feature_file_sha256": feature_sha256,
                        "checkpoint": checkpoint,
                        "preprocess_sha256": preprocess_sha256,
                        "processing": processing,
                    })
                    write_json(metadata_file, metadata)
                    cache_valid = True
                    print(f"upgraded verified cache metadata for {shard_name}", flush=True)
            except Exception:
                cache_valid = False
            if cache_valid:
                for index, row in enumerate(shard_rows):
                    index_rows.append({**row, "feature_file": feature_file.relative_to(root).as_posix(), "feature_row": index, "feature_config_hash": cfg_hash})
                print(f"cached {shard_name}", flush=True)
                continue

        embeddings: list[np.ndarray] = []
        successful_rows: list[dict[str, str]] = []
        current_batch_size = batch_size
        start = 0
        while start < len(shard_rows):
            next_start, tensors, usable_rows, batch_failures = load_image_batch(
                shard_rows, start, current_batch_size, root, preprocess
            )
            if not tensors:
                failures.extend(batch_failures)
                start = next_start
                continue
            try:
                images = torch.stack(tensors).to(device)
                with torch.inference_mode():
                    features = model.encode_image(images)
                    if normalize:
                        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            except torch.cuda.OutOfMemoryError:
                if device != "cuda" or current_batch_size <= 1:
                    raise
                del tensors
                if "images" in locals():
                    del images
                torch.cuda.empty_cache()
                current_batch_size = max(1, current_batch_size // 2)
                print(f"CUDA OOM; retrying {shard_name} with batch_size={current_batch_size}", flush=True)
                continue
            failures.extend(batch_failures)
            embeddings.append(features.float().cpu().numpy())
            successful_rows.extend(usable_rows)
            start = next_start
        matrix = np.concatenate(embeddings, axis=0).astype(np.float32, copy=False) if embeddings else np.empty((0, 0), dtype=np.float32)
        image_ids = np.asarray([row["image_id"] for row in successful_rows])
        temporary_feature_file = feature_file.with_suffix(".tmp.npz")
        np.savez_compressed(temporary_feature_file, image_ids=image_ids, embeddings=matrix)
        temporary_feature_file.replace(feature_file)
        feature_sha256 = file_sha256(feature_file)
        cache_spec = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "model_name": model_name,
            "pretrained": pretrained,
            "checkpoint": checkpoint,
            "preprocess_sha256": preprocess_sha256,
            "processing": processing,
            "variant": args.variant,
            "l2_normalize": normalize,
            "embedding_dimension": expected_dimension,
            "input_fingerprint": inputs_sha256,
        }
        metadata = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "cache_spec_hash": canonical_hash(cache_spec),
            "config_hash": cfg_hash,
            "model_name": model_name,
            "pretrained": pretrained,
            "variant": args.variant,
            "device": device,
            "requested_batch_size": batch_size,
            "effective_batch_size": current_batch_size,
            "l2_normalize": normalize,
            "checkpoint_cache_dir": str(cache_dir),
            "checkpoint": checkpoint,
            "preprocess_sha256": preprocess_sha256,
            "processing": processing,
            "input_fingerprint": inputs_sha256,
            "feature_file_sha256": feature_sha256,
            "embedding_dimension": int(matrix.shape[1]) if matrix.ndim == 2 and matrix.size else 0,
            "image_ids": image_ids.tolist(),
        }
        write_json(metadata_file, metadata)
        for index, row in enumerate(successful_rows):
            index_rows.append({**row, "feature_file": feature_file.relative_to(root).as_posix(), "feature_row": index, "feature_config_hash": cfg_hash})
        print(f"extracted {shard_name}: {len(successful_rows)}", flush=True)

    if index_rows:
        write_csv(output_root / "feature_index.csv", index_rows, list(index_rows[0].keys()))
    if failures:
        write_csv(output_root / "failures.csv", failures, ["image_id", "error"])
    write_json(output_root / "extraction_settings.json", {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "config_hash": cfg_hash, "model_name": model_name, "pretrained": pretrained,
        "variant": args.variant, "device": device, "requested_batch_size": batch_size,
        "l2_normalize": normalize, "successful": len(index_rows), "failed": len(failures),
        "checkpoint_cache_dir": str(cache_dir),
        "checkpoint": checkpoint,
        "preprocess_sha256": preprocess_sha256,
        "processing": processing,
    })
    print(f"successful={len(index_rows)} failed={len(failures)} output={output_root}")


if __name__ == "__main__":
    main()
