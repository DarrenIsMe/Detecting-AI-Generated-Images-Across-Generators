from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

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

    cfg_hash = config_hash(model_name, pretrained, args.variant, normalize)
    output_root = root / "data" / "features" / args.variant / f"{model_name}-{pretrained}-{cfg_hash}"
    output_root.mkdir(parents=True, exist_ok=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device, cache_dir=str(cache_dir)
    )
    model.eval()
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    by_shard: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_shard.setdefault(Path(row["shard"]).stem, []).append(row)

    index_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for shard_name, shard_rows in sorted(by_shard.items()):
        feature_file = output_root / f"{shard_name}.npz"
        metadata_file = output_root / f"{shard_name}.json"
        expected_ids = [row["image_id"] for row in shard_rows]
        if feature_file.exists() and metadata_file.exists():
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            try:
                with np.load(feature_file) as archive:
                    cached_ids = archive["image_ids"].astype(str).tolist()
                    cached_embeddings = archive["embeddings"]
                    cache_valid = (
                        metadata.get("config_hash") == cfg_hash
                        and metadata.get("image_ids") == expected_ids
                        and cached_ids == expected_ids
                        and cached_embeddings.shape[0] == len(expected_ids)
                        and cached_embeddings.ndim == 2
                        and cached_embeddings.shape[1] == 512
                        and np.isfinite(cached_embeddings).all()
                    )
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
            batch_rows = shard_rows[start:start + current_batch_size]
            tensors = []
            usable_rows = []
            for row in batch_rows:
                try:
                    path = root / row["image_path"]
                    with Image.open(path) as image:
                        tensors.append(preprocess(image.convert("RGB")))
                    usable_rows.append(row)
                except Exception as exc:
                    failures.append({"image_id": row["image_id"], "error": f"{type(exc).__name__}: {exc}"})
            if not tensors:
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
            embeddings.append(features.float().cpu().numpy())
            successful_rows.extend(usable_rows)
            start += len(batch_rows)
        matrix = np.concatenate(embeddings, axis=0).astype(np.float32, copy=False) if embeddings else np.empty((0, 0), dtype=np.float32)
        image_ids = np.asarray([row["image_id"] for row in successful_rows])
        temporary_feature_file = feature_file.with_suffix(".tmp.npz")
        np.savez_compressed(temporary_feature_file, image_ids=image_ids, embeddings=matrix)
        temporary_feature_file.replace(feature_file)
        metadata = {
            "config_hash": cfg_hash,
            "model_name": model_name,
            "pretrained": pretrained,
            "variant": args.variant,
            "device": device,
            "requested_batch_size": batch_size,
            "effective_batch_size": current_batch_size,
            "l2_normalize": normalize,
            "checkpoint_cache_dir": str(cache_dir),
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
        "config_hash": cfg_hash, "model_name": model_name, "pretrained": pretrained,
        "variant": args.variant, "device": device, "requested_batch_size": batch_size,
        "l2_normalize": normalize, "successful": len(index_rows), "failed": len(failures),
        "checkpoint_cache_dir": str(cache_dir),
    })
    print(f"successful={len(index_rows)} failed={len(failures)} output={output_root}")


if __name__ == "__main__":
    main()
