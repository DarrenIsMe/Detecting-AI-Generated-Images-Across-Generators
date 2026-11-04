# Handoff for Person 2 and Person 3

## Stable identifiers and joins

Use `image_id` as the only join key. Do not join by filename. Every processed manifest and feature index contains all 35,000 unique image IDs. Feature rows must be loaded using both `feature_file` and `feature_row`; verify that the archive's `image_ids[feature_row]` equals the CSV `image_id`.

## Person 2: model experiments

1. Select one CSV from `data/splits/` and keep its `split` assignments unchanged.
2. Join it to one condition's `feature_index.csv`, or to the matching processed manifest for image training.
3. Never use the held-out generator in training or validation. For strict transfer, use `biggan_to_diffusion_strict.csv` and report `biggan_to_midjourney_separate_test.csv` separately.
4. Preserve `duplicate_group_id` assignments. Do not re-split individual images.
5. Use `label` (`0=real`, `1=fake`), `generator`, `generator_family`, `fold`, and `evaluation_group` directly.

Recommended default input condition: `standardized`. Always compare with `original_encoded`; treat `reencode_only` and `jpeg_common` as controls, not extra training data.

## Person 3: robustness and attacks

1. Start from test rows only in the chosen split manifest.
2. Resolve clean input paths through the matching condition manifest in `data/manifests/`.
3. Record every transformation and retain the original `image_id`.
4. Re-extract transformed images with the checkpoint settings in `results/clip_model/CLIP_CONFIG.md`.
5. Never mix transformed test samples into training or validation.

## Feature directories

Each condition contains a single configuration directory under:

- `data/features/original_encoded/`
- `data/features/standardized/`
- `data/features/reencode_only/`
- `data/features/jpeg_common/`

Within it, use `feature_index.csv`, `extraction_settings.json`, and the 18 per-shard `.npz` archives. The extractor binds caches to exact input bytes, preprocessing settings, checkpoint hashes, and feature-file hashes; it recomputes an invalid or partial shard.

## Reproduction sequence

From PowerShell in the project root:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\inspect_datasets.py
python scripts\make_splits.py
python scripts\validate_pipeline.py --master-manifest data\manifests\master_manifest.csv
python scripts\prepare_images.py standardized
python scripts\extract_clip.py standardized
python scripts\format_ablation.py `
  --split-manifest data\splits\biggan_to_diffusion_strict.csv `
  --additional-test-manifest data\splits\biggan_to_midjourney_separate_test.csv `
  --feature-index data\features\standardized\ViT-B-32-quickgelu-openai-e8d939fb5c8e47f6\feature_index.csv `
  --variant standardized
python scripts\generate_final_report.py
```

The preparation and extraction commands are resumable and skip verified completed outputs. Raw Parquet files and the original DeepTrust ZIP are never modified.

## Required caveats in reports

- Midjourney architecture is proprietary/unknown, not confirmed diffusion.
- BigGAN is the only GAN represented.
- All original real images are JPEG and all original fake images are PNG.
- Common JPEG re-encoding does not remove artifacts already introduced by an earlier JPEG encoding.
- Tiny-GenImage does not provide explicit fake-to-real source pairing, so some source-level leakage cannot be ruled out.
