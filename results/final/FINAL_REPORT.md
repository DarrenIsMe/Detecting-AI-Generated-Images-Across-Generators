# Dataset Pipeline and Feature Report

I completed and validated the dataset, preprocessing, split-generation, and feature-extraction pipeline. Overall validation: **ok**.

## Completed outputs

| Condition | Manifest rows | Feature rows | Processed size (GiB) |
|---|---:|---:|---:|
| `original_encoded` | 35,000 | 35,000 | 7.784 |
| `standardized` | 35,000 | 35,000 | 2.787 |
| `reencode_only` | 35,000 | 35,000 | 11.763 |
| `jpeg_common` | 35,000 | 35,000 | 3.719 |

All four conditions cover every existing experiment manifest. Every saved feature archive is 512-dimensional float32, finite, L2-normalized, and ordered by explicit image ID.

## Strict BigGAN-to-diffusion ablation (seed 4001)

| Condition | Validation accuracy | Diffusion test accuracy | Diffusion test F1 | Midjourney accuracy (separate) |
|---|---:|---:|---:|---:|
| `original_encoded` | 0.9860 | 0.61272 | 0.39130 | 0.5150 |
| `standardized` | 0.9860 | 0.61272 | 0.39122 | 0.5150 |
| `reencode_only` | 0.9860 | 0.61272 | 0.39130 | 0.5150 |
| `jpeg_common` | 0.9780 | 0.55176 | 0.22951 | 0.4910 |

Seeds 4001, 4002, and 4003 produced identical metrics because the fixed split and convex linear classifier converged to the same solution.

The common-JPEG control reduces strict test accuracy and fake recall. This shows that codec treatment materially affects the detector, but it does not prove that JPEG artifacts are removed: JPEG sources are decoded with their existing artifacts and then compressed again.

Midjourney is treated as proprietary/unknown and is not included in the strict diffusion aggregate. It is never used for training or validation.

## Runtime summary

| Stage | Seconds | Minutes |
|---|---:|---:|
| `clip_smoke_32_images` | 14.775 | 0.25 |
| `preprocess_original_encoded` | 245.783 | 4.10 |
| `preprocess_standardized` | 591.358 | 9.86 |
| `preprocess_reencode_only` | 1356.005 | 22.60 |
| `preprocess_jpeg_common` | 301.518 | 5.03 |
| `clip_original_encoded` | 618.966 | 10.32 |
| `clip_standardized` | 417.111 | 6.95 |
| `clip_reencode_only` | 743.271 | 12.39 |
| `clip_jpeg_common` | 564.934 | 9.42 |
| `ablation_original_encoded` | 13.303 | 0.22 |
| `ablation_standardized` | 4.577 | 0.08 |
| `ablation_reencode_only` | 4.818 | 0.08 |
| `ablation_jpeg_common` | 4.584 | 0.08 |

## Important limitations

- BigGAN is the only GAN, so the strict protocol is BigGAN-to-diffusion rather than broad GAN-family generalization.
- Tiny-GenImage does not document its sampling method from full GenImage and does not retain explicit fake-to-real source pairing.
- Exact duplicates are grouped atomically; reviewed perceptual-hash candidates were false positives. Undetected semantic/source reuse remains an unresolved risk.
- Original real images are JPEG and original fake images are PNG, a severe format confound.
