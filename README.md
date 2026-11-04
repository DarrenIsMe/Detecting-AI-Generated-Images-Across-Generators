# Detecting AI-Generated Images Across Generators

NTU SC4001 Neural Networks and Deep Learning group project investigating whether detectors trained on one image generator generalize to unseen generators. This branch contains Person 1's completed dataset audit, leakage-safe split generation, preprocessing controls, frozen CLIP feature extraction, and input-standardization ablation. These ablation numbers are not the group's final main-classifier results.

## Dataset and provenance

The pipeline uses [Tiny-GenImage](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage), a 35,000-image subset of [GenImage](https://github.com/GenImage-Dataset/GenImage), pinned to revision `89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69`.

- 17,500 real and 17,500 generated images.
- Seven available fake generators: ADM, BigGAN, GLIDE, Midjourney, SD 1.5, VQDM, and Wukong; SD 1.4 is in the vocabulary but absent.
- Midjourney is treated as proprietary/unknown architecture, not confirmed diffusion.
- Licence: CC BY-NC-SA 4.0; use is non-commercial and attribution/share-alike obligations apply.
- Full audit: 35,000 readable images, no corrupt files, and two exact duplicate pairs. Five perceptual-hash candidates were manually reviewed as false positives.
- All original real images are JPEG and all original fake images are PNG, creating a severe format confound.

See [the audit summary](results/dataset_statistics/AUDIT_SUMMARY.md) and [manual duplicate review](results/dataset_statistics/PERCEPTUAL_REVIEW.md).

## Repository contents

- `scripts/`: audit, preprocessing, split, CLIP, validation, and ablation programs.
- `config/datasets.yaml`: pinned dataset, generator families, processing, split, and CLIP settings.
- `data/splits/`: lean exports of the completed generator-aware assignments. Images and embeddings are not included.
- `results/`: genuine completed audit, ablation, validation, and handoff outputs.
- `tests/`: lightweight unit tests that do not require the dataset or model weights.

## Installation

Python 3.12 was used for the completed run.

```bash
git clone https://github.com/DarrenIsMe/Detecting-AI-Generated-Images-Across-Generators.git
cd Detecting-AI-Generated-Images-Across-Generators
python -m venv .venv
```

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The recorded GPU environment used PyTorch `2.11.0+cu128`, torchvision `0.26.0+cu128`, OpenCLIP `3.3.0`, and an NVIDIA GTX 1660 SUPER with 6 GB VRAM. On CPU-only systems, install compatible CPU PyTorch/torchvision wheels instead; extraction automatically falls back to CPU.

## Dataset setup

Datasets, images, embeddings, checkpoints, and caches are intentionally excluded from Git. Download the pinned Tiny-GenImage revision into the default location:

```powershell
hf download TheKernel01/Tiny-GenImage `
  --repo-type dataset `
  --revision 89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69 `
  --local-dir data\raw\tiny_genimage_hf
```

To reuse an existing copy without duplicating it, point the pipeline at that directory:

```powershell
$env:SC4001_TINY_GENIMAGE_DIR = 'D:\path\to\tiny_genimage_hf'
$env:SC4001_CLIP_CACHE_DIR = 'D:\path\to\open_clip_cache' # optional
```

Bash equivalents are `export SC4001_TINY_GENIMAGE_DIR=...` and `export SC4001_CLIP_CACHE_DIR=...`. An absolute `dataset.raw_dir` in `config/datasets.yaml` also works, but environment variables keep local paths out of commits.

## Audit and leakage-safe splits

```powershell
# Quick audit smoke test
python scripts\inspect_datasets.py --limit 200

# Full audit and master manifest
python scripts\inspect_datasets.py

# Regenerate exact duplicate-aware protocol assignments
python scripts\make_splits.py
python scripts\validate_pipeline.py
```

Duplicate groups are atomic and take priority over exact sample counts. Real-image train, validation, and test partitions are disjoint. A held-out generator is never present in training or validation.

Protocols:

- Seven leave-one-generator-out files: `logo_<generator>.csv`.
- Strict transfer: train/validate on BigGAN and matched real images, then test on ADM, GLIDE, SD 1.5, VQDM, and Wukong.
- Midjourney is an evaluation-only manifest reported separately from strict diffusion results.

The committed CSVs are lean exports. Regenerating splits after the full audit restores additional source metadata without changing the completed assignments for revision and seed 4001.

## Preprocessing controls

```powershell
python scripts\prepare_images.py original_encoded
python scripts\prepare_images.py standardized
python scripts\prepare_images.py reencode_only
python scripts\prepare_images.py jpeg_common
```

Conditions:

1. `original_encoded`: exact bytes distributed in Tiny-GenImage.
2. `standardized`: RGB, bicubic shorter-side resize, 224×224 centre crop, lossless PNG.
3. `reencode_only`: RGB and lossless PNG while preserving dimensions.
4. `jpeg_common`: baseline JPEG quality 95, 4:4:4 chroma, preserving dimensions.

Common JPEG re-encoding makes the final codec consistent but cannot remove JPEG artifacts already present in the real source images. Preparation is resumable and skips existing outputs unless `--overwrite` is supplied.

## Frozen CLIP extraction

The completed configuration is OpenAI CLIP ViT-B/32 with QuickGELU:

- OpenCLIP name: `ViT-B-32-quickgelu`
- Pretrained tag: `openai`
- 512-dimensional float32, L2-normalized embeddings
- Checkpoint SHA-256: `E6D1BD7789AA45192B3BF90570A789B478BAE1B74EBCCE7EDDD908E83A2B7C31`

```powershell
python scripts\extract_clip.py original_encoded
python scripts\extract_clip.py standardized
python scripts\extract_clip.py reencode_only
python scripts\extract_clip.py jpeg_common
```

The extractor uses CUDA when available, falls back to CPU, and halves the batch size automatically after a CUDA out-of-memory error. It stores and validates one atomic cache per source shard, so completed shards are not recomputed.

### Loading features safely

Each condition produces a `feature_index.csv`. Join it to a split using `image_id`, then load both `feature_file` and `feature_row`. Always verify the stored ID:

```python
import csv
import numpy as np
from pathlib import Path

root = Path(".")
rows = list(csv.DictReader(open("data/features/standardized/<config>/feature_index.csv")))
row = rows[0]
with np.load(root / row["feature_file"]) as archive:
    index = int(row["feature_row"])
    assert str(archive["image_ids"][index]) == row["image_id"]
    vector = archive["embeddings"][index]  # shape: (512,)
```

Images and embeddings must be shared separately through team-approved storage; do not add them to Git. Teammates can reproduce them from the pinned dataset and configuration above.

## Person 1 input-standardization ablation

A balanced logistic regression (`C=1`, `class_weight="balanced"`, maximum 2,000 iterations) used identical strict splits and classifier settings for all conditions. Midjourney was never used for training or validation.

| Condition | Validation accuracy | Strict diffusion test accuracy | Strict test F1 | Midjourney accuracy, separate |
|---|---:|---:|---:|---:|
| Original encoded | 98.600% | 61.272% | 39.130% | 51.500% |
| Standardized 224×224 PNG | 98.600% | 61.272% | 39.122% | 51.500% |
| Lossless re-encoded PNG | 98.600% | 61.272% | 39.130% | 51.500% |
| Common JPEG Q95 | 97.800% | 55.176% | 22.951% | 49.100% |

These are Person 1 format-control results, not final project classifier performance. Detailed per-generator metrics and confusion matrices are under `results/input_standardization_ablation/`.

The completed consistency validation reloaded all 35,000 features for every condition, verified 512 dimensions, exact feature/image-ID ordering, full manifest and split coverage, and absence of NaN/Inf values. Original, standardized, and lossless re-encoded features also passed the requested feature/prediction consistency checks; common JPEG passed the same feature validation.

Reproduce one condition:

```powershell
python scripts\format_ablation.py `
  --split-manifest data\splits\biggan_to_diffusion_strict.csv `
  --additional-test-manifest data\splits\biggan_to_midjourney_separate_test.csv `
  --feature-index data\features\standardized\<config>\feature_index.csv `
  --variant standardized
```

## Teammate handoff

Person 2 should preserve the committed `split` assignments and join features by `image_id`. The manifests expose binary labels, generator identities/families, duplicate groups, protocol folds, and evaluation groups for GAN-to-diffusion and leave-one-generator-out experiments.

Person 3 should use only test rows, retain `image_id` through every JPEG/resizing transformation, and re-extract transformed features with the same CLIP configuration. The locally generated master manifest supplies Parquet shard/row locations and original filenames for failure analysis.

See [the detailed Person 2/3 handoff](results/final/HANDOFF_PERSON2_PERSON3.md).

## Validation

```powershell
python -m pytest -q
python scripts\validate_pipeline.py
```

The completed full run reported four passing unit tests and zero manifest, split, duplicate-group, held-generator, feature-ordering, dimensionality, or numerical validation errors.

## Known limitations

- Tiny-GenImage does not document its sampling procedure from full GenImage.
- It provides no explicit generated-to-real source pairing, so undetected source reuse cannot be ruled out.
- BigGAN is the only GAN; strict transfer is BigGAN-to-diffusion, not broad GAN-family generalization.
- SD 1.4 is absent.
- Original encoding is severely class-confounded: all real images are JPEG and all fake images are PNG.
- `original_encoded` refers to Tiny-GenImage's distributed bytes, not proven byte-identical copies from full GenImage.

## AI-use disclosure

Generative AI tools assisted with code drafting, debugging, documentation, and reproducibility checks. The student reviewed the implementation, ran the reported tests and experiments, and remains responsible for the submitted work. The team should include this disclosure, along with any course-mandated tool names or interaction records, in the final NTU SC4001 submission according to the current course requirements.
