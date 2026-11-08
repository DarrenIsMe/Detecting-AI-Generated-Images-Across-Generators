# Detecting AI-Generated Images Across Generators

NTU SC4001 Neural Networks and Deep Learning group project investigating whether detectors trained on one image generator generalize to unseen generators. I implemented the dataset audit, leakage-safe split generation, preprocessing controls, frozen CLIP feature extraction, and input-standardization ablation documented on this branch. These ablation numbers are not the group's final main-classifier results.

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
python scripts\validate_pipeline.py --master-manifest data\manifests\master_manifest.csv
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

The extractor uses CUDA when available, falls back to CPU, and halves the batch size automatically after a CUDA out-of-memory error. It stores one atomic cache per source shard. Cache reuse verifies ordered IDs, dimensions, finite values, exact feature-file bytes, exact processed-image bytes, preprocessing settings, the OpenCLIP transform, and checkpoint hashes. A committed attestation allows only the previously verified baseline caches to receive upgraded metadata without re-extraction; any mismatch is recomputed.

## Downloading Precomputed CLIP Features

The [Precomputed CLIP Features v1 release](https://github.com/DarrenIsMe/Detecting-AI-Generated-Images-Across-Generators/releases/tag/person1-features-v1) provides one ZIP per preprocessing condition. The archives contain NPZ feature arrays, `feature_index.csv`, sanitized extraction metadata, and attribution/licence notices. They do not contain source or processed images.

### Option A — use the release assets

Download the required condition and extract it into the repository root. For example, the standardized features can be used for model experiments without repeating preprocessing or CLIP inference:

```powershell
Invoke-WebRequest `
  https://github.com/DarrenIsMe/Detecting-AI-Generated-Images-Across-Generators/releases/download/person1-features-v1/clip_features_standardized.zip `
  -OutFile clip_features_standardized.zip
Expand-Archive .\clip_features_standardized.zip -DestinationPath . -Force
```

Extraction creates:

```text
data/features/standardized/ViT-B-32-quickgelu-openai-e8d939fb5c8e47f6/
```

Available assets and SHA-256 checksums:

| Condition | Asset | SHA-256 |
|---|---|---|
| Original encoded | `clip_features_original_encoded.zip` | `6b9d5c19eb2eb8d8000e27a10277e204f40a9d00ef7a1b89718ab92c674a996d` |
| Standardized | `clip_features_standardized.zip` | `f74bde073d33433a9c37cbbb452eb8c706353714430c13461df00453da57bfb7` |
| Lossless re-encoded | `clip_features_reencode_only.zip` | `991da95ada45dccfdcc97e875ecc6fc584a1c0a0e6d5a75fcb04b7a7953cf75b` |
| Common JPEG Q95 | `clip_features_jpeg_common.zip` | `a5240dc8671bb5c62731047a32044c6d05cea71fc2d2fc094a64711fb93270ea` |

Verify a download before extraction:

```powershell
(Get-FileHash .\clip_features_standardized.zip -Algorithm SHA256).Hash
```

The teammate responsible for model experiments can join `feature_index.csv` to any committed split manifest using `image_id` and train a linear classifier directly. For the completed strict experiment:

```powershell
python scripts\format_ablation.py `
  --split-manifest data\splits\biggan_to_diffusion_strict.csv `
  --additional-test-manifest data\splits\biggan_to_midjourney_separate_test.csv `
  --feature-index data\features\standardized\ViT-B-32-quickgelu-openai-e8d939fb5c8e47f6\feature_index.csv `
  --variant standardized
```

### Option B — reproduce features from Tiny-GenImage

Install the Hugging Face CLI and download the pinned dataset revision:

```powershell
python -m pip install --upgrade huggingface_hub
hf download TheKernel01/Tiny-GenImage `
  --repo-type dataset `
  --revision 89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69 `
  --local-dir data\raw\tiny_genimage_hf
python scripts\inspect_datasets.py
python scripts\make_splits.py
python scripts\prepare_images.py standardized
python scripts\extract_clip.py standardized
```

The CNN baseline still requires original or reproducibly processed images. The teammate responsible for robustness experiments needs original or processed images to create new JPEG/resizing conditions. Image data is not included in the release. Regenerate `data/manifests/master_manifest.csv` with `inspect_datasets.py` when source Parquet rows, filenames, or image paths are required; generated manifests must not rely on another user's absolute paths.

The release features are derived from Tiny-GenImage/GenImage and are distributed conservatively under CC BY-NC-SA 4.0 for non-commercial research and teaching, with attribution and share-alike requirements. Review the licence notice inside each archive.

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

## Input-Standardization Ablation

I ran a balanced logistic regression (`C=1`, `class_weight="balanced"`, maximum 2,000 iterations) with identical strict splits and classifier settings for all conditions. Midjourney was never used for training or validation.

| Condition | Validation accuracy | Strict diffusion test accuracy | Strict test F1 | Midjourney accuracy, separate |
|---|---:|---:|---:|---:|
| Original encoded | 98.600% | 61.272% | 39.130% | 51.500% |
| Standardized 224×224 PNG | 98.600% | 61.272% | 39.122% | 51.500% |
| Lossless re-encoded PNG | 98.600% | 61.272% | 39.130% | 51.500% |
| Common JPEG Q95 | 97.800% | 55.176% | 22.951% | 49.100% |

These are my format-control results, not the final project classifier performance. Detailed per-generator metrics and confusion matrices are under `results/input_standardization_ablation/`.

The completed consistency validation reloaded all 35,000 features for every condition and verified 512 dimensions, exact feature/image-ID ordering, full manifest and split coverage, and absence of NaN/Inf values. The ablation regenerated classifier predictions from those validated features for all four conditions. No separate per-image prediction-consistency artifact was saved, so the repository does not claim one.

Reproduce one condition:

```powershell
python scripts\format_ablation.py `
  --split-manifest data\splits\biggan_to_diffusion_strict.csv `
  --additional-test-manifest data\splits\biggan_to_midjourney_separate_test.csv `
  --feature-index data\features\standardized\<config>\feature_index.csv `
  --variant standardized
```

## Teammate handoff

The teammate responsible for model experiments should preserve the committed `split` assignments and join features by `image_id`. The manifests expose binary labels, generator identities/families, duplicate groups, protocol folds, and evaluation groups for GAN-to-diffusion and leave-one-generator-out experiments.

The teammate responsible for robustness experiments should use only test rows, retain `image_id` through every JPEG/resizing transformation, and re-extract transformed features with the same CLIP configuration. The locally generated master manifest supplies Parquet shard/row locations and original filenames for failure analysis.

See [the detailed teammate handoff](results/final/HANDOFF_PERSON2_PERSON3.md).

## Validation

```powershell
# Dataset-independent lightweight tests
python -m pytest -q

# Requires master_manifest.csv generated by inspect_datasets.py
python scripts\validate_pipeline.py --master-manifest data\manifests\master_manifest.csv
```

The current lightweight suite reports 12 passing tests. The completed full-data validation reported zero manifest, split, duplicate-group, held-generator, feature-ordering, dimensionality, or numerical errors.

## Known limitations

- Tiny-GenImage does not document its sampling procedure from full GenImage.
- It provides no explicit generated-to-real source pairing, so undetected source reuse cannot be ruled out.
- BigGAN is the only GAN; strict transfer is BigGAN-to-diffusion, not broad GAN-family generalization.
- SD 1.4 is absent.
- Original encoding is severely class-confounded: all real images are JPEG and all fake images are PNG.
- `original_encoded` refers to Tiny-GenImage's distributed bytes, not proven byte-identical copies from full GenImage.

## Part 2: main experiments

Part 2 trains a logistic-regression head on frozen features (OpenCLIP ViT-B/32 and ViT-L/14, torchvision ResNet-50) under two preprocessing pipelines (A: resize 224 + centre crop; B: native 112x112 crop resized to 224) and two formats (`raw`, `matched_jpeg`). Part 1's files, splits and caches are not modified. Settings live in `config/part2.yaml`; code is in `scripts/part2/` with CLIs `scripts/part2_*.py`.

### Setup

Follow Installation and Dataset setup above, with the pinned dataset revision and all commands run from the repo root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt huggingface_hub
.\.venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 2.11.0+cu128 True
.\.venv\Scripts\python -m pytest -q                                                              # unit tests
.\.venv\Scripts\hf download TheKernel01/Tiny-GenImage --repo-type dataset --revision 89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69 --local-dir data\raw\tiny_genimage_hf
.\.venv\Scripts\python scripts\inspect_datasets.py       # regenerates data\manifests\master_manifest.csv
git restore results/dataset_statistics                   # keep Part 1's committed audit files
.\.venv\Scripts\python scripts\validate_pipeline.py --master-manifest data\manifests\master_manifest.csv
```

`py` below means `.\.venv\Scripts\python`.

### Commands, in order

```powershell
py scripts\part2_jpeg_profile.py             # modal JPEG quality/subsampling of training reals -> results\part2\jpeg_profile.json
py scripts\part2_extract.py --all            # 12 feature sets (3 backbones x A/B x raw/matched_jpeg), clip_l14 first -> data\part2\features\
py scripts\part2_run.py --reproduce          # gate: must reproduce Part 1's CLIP B/32 baseline (status "ok") before anything else is trusted
py scripts\part2_run.py --grid               # strict + logo (full fit + seeds 4001-4003), cross-condition pairs, final classifiers
py scripts\part2_tables.py                   # the report tables (prints WARNING lines for incomplete conditions)
py scripts\part2_predict.py --image PATH --head results\part2\heads\clip_l14__A__raw__final__all__full.json   # one image -> {"p_fake": ..., "label": ...}
```

The matched-JPEG recipe is the mode over the union of the strict-train and LOGO-train reals; `part2_jpeg_profile.py` also records each protocol's own profile under `per_protocol` and stops if their modes differ.

A single configuration can be run with `part2_run.py --backbone B --pipeline P --format F --protocol {strict,logo,final} [--test-pipeline P2 --test-format F2] [--no-seeds]`; finished runs are skipped unless `--overwrite` (see "Cache and result reuse").

### Preprocessing and numerics

- Images are decoded with Pillow and converted to RGB. There is no EXIF transpose (as in Part 1's `original_encoded` path): an image with an EXIF orientation tag is used as stored.
- Pipeline geometry (A: size 224; B: crop 112, grid 16, size 224) and the 112 px floor are hard-coded in `scripts/part2/preprocess.py` and repeated in `config/part2.yaml`, which is what feature specs and heads record. `part2_extract.py`, `part2_run.py` and `score_images` refuse to start if the two disagree.
- Embeddings are computed in fp32, but PyTorch's default `torch.backends.cudnn.allow_tf32 = True` is left on, so on the RTX 4070 the convolutions (all of ResNet-50, and the patch-embedding convolution of both CLIP models) run in TF32. Matrix multiplications (attention, MLP and projection layers) stay in full fp32, since `torch.backends.cuda.matmul.allow_tf32` defaults to `False`. Features re-extracted on another GPU, or on the CPU, can therefore differ slightly.

### Cache and result reuse

- A feature cache is reused when its spec hash matches: backbone and pipeline config blocks, JPEG recipe, and the ordered image ids. The hash does not cover the code, so **after any change to preprocessing or extraction code, re-extract with `part2_extract.py --overwrite`**. `part2_run.py` recomputes the hash from the current `config/part2.yaml` and refuses a cache built with other settings.
- Every run records a fingerprint in its head and in every metric row: the id, spec hash and `created_at` of its train and test feature sets, plus `C_grid`, `max_iter`, `subsample_fraction`, `min_side` and the seed. `part2_run.py` skips a run only when its outputs exist with the current fingerprint; otherwise it prints `stale <run_id>: re-running` and fits it again. Re-extracting features or changing those settings therefore never leaves old metrics in the tables.

### Where outputs live

| Output | Location |
|---|---|
| Metric rows (one JSON list per run) | `results/part2/metrics/<run_id>.json` |
| Full-data heads | `results/part2/heads/<run_id>.json` |
| Reproduction check, exclusion counts | `results/part2/reproduction.json`, `results/part2/exclusions.json` |
| Report tables (Markdown + CSV) | `results/part2/tables/` |
| Features, seed heads, predictions (gitignored) | `data/part2/features/`, `data/part2/heads/`, `data/part2/predictions/` |

Only the report tables and `jpeg_profile.json` are committed; the metric rows, heads, reproduction and exclusion files are regenerated by the commands above.

### Report tables

`part2_tables.py` writes five tables (Markdown and CSV) to `results/part2/tables/`:

- `strict_overall` (per condition) and `strict_per_generator` (per condition and strict test generator; Midjourney has `in_overall = False`) give each metric as the deterministic full-data fit (`<metric>_full`) and the mean and sd (ddof=1) over seeds 4001-4003 (`<metric>_mean`, `<metric>_sd`), with `n_seeds`.
- `logo_per_fold` gives the same per held-out generator, plus one `mean_over_folds` row per condition: each fit's mean over the 7 folds, then the mean and sd of those means across seeds, next to the full-data fold mean. `n_folds` is the number of folds a row averages.
- `cross_condition` uses full-data heads only (cross-condition runs are not seeded). It shows `<metric>_same`, the head scored on its own training condition, next to `<metric>_cross`, the same head on the test condition, so the transfer drop is `_same - _cross`; LOGO pairs are averaged over folds.
- `final_validation` is headed "in-distribution only, not generalisation".

The oracle balanced accuracy appears as `oracle_bacc_upper_bound`, because its threshold is picked on the test labels. The script prints a `WARNING:` line for every condition with fewer than 3 seeds or 7 LOGO folds, and for metric rows that mix feature extractions or settings.

### The final classifier

The `final` protocol trains on all seven generators. Its train and validation sets are the union of the seven LOGO folds' train and validation rows, de-duplicated by `image_id`; any duplicate group present in both is dropped from both. The canonical single-image head is `results/part2/heads/clip_l14__A__raw__final__all__full.json` (ViT-L/14, standard CLIP preprocessing, raw files). The other three final heads (`clip_l14__A__matched_jpeg__…`, `clip_l14__B__raw__…`, `clip_l14__B__matched_jpeg__…`) are comparisons.

### Scoring API for Part 3

The head's metadata selects the backbone and pipeline. Load the backbone once and pass it to every call:

```python
from pathlib import Path
from PIL import Image
from common import load_config                     # scripts/ must be on sys.path
from part2.backbones import load_backbone
from part2.scoring import score_images

root = Path.cwd()                                  # the repo root
cfg = load_config(root, "config/part2.yaml")
head = root / "results/part2/heads/clip_l14__A__raw__final__all__full.json"
backbone = load_backbone("clip_l14", cfg, root)    # slow: load once, reuse for every call
p_fake = score_images([Image.open("example.png")], head, backbone=backbone)   # numpy array of P(fake)
```

`embed_images(images, backbone, pipeline)` returns the L2-normalised embeddings. Before scoring, `score_images` checks the feature-extraction settings recorded in the head (backbone and pipeline config blocks, normalisation) against `config/part2.yaml` and against the backbone's name, dimension and normalisation. It raises `ValueError` on any mismatch, and for heads written before these settings were recorded (re-run `part2_run.py` with `--overwrite` to regenerate them). It raises `ImageTooSmall` for images below the minimum size of the head's pipeline.

No format conversion happens at inference: images are scored as given. A `matched_jpeg` head was trained on fakes re-encoded as JPEG, so scoring it in its own condition means re-encoding the fakes first (reals are never re-encoded):

```python
from part2.formats import load_condition_image, load_jpeg_spec

spec = load_jpeg_spec(Path("results/part2/jpeg_profile.json"))           # also in the head's meta["jpeg"]
image = load_condition_image(encoded_bytes, label, "matched_jpeg", spec)  # label 1: re-encoded; label 0: unchanged
```

Scoring the original PNG fakes with a `matched_jpeg` head is a cross-condition test.

### Caveats

- Images whose shorter side is under 112 px are excluded from every condition and protocol (counts per split in `results/part2/exclusions.json`); only the reproduction check keeps them.
- `matched_jpeg` re-encodes fakes only, at the training reals' modal JPEG quality and subsampling; reals are never re-encoded. It reduces, but does not prove away, the real-JPEG versus fake-PNG confound.
- Oracle balanced accuracy (`oracle_bacc_upper_bound` in the tables) uses the best threshold chosen on the test labels, so it is an upper bound, not a result.
- The `final` classifier is trained on all 7 generators and is for the single-image interface only. Its validation numbers are in-distribution and are not evidence of generalisation; cross-generator evidence comes from the `strict` and `logo` protocols.

## AI-use disclosure

Generative AI tools assisted with code drafting, debugging, documentation, and reproducibility checks. The student reviewed the implementation, ran the reported tests and experiments, and remains responsible for the submitted work. The team should include this disclosure, along with any course-mandated tool names or interaction records, in the final NTU SC4001 submission according to the current course requirements.
