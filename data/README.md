# Data interface

This repository intentionally excludes raw images, processed images, CLIP embeddings, model checkpoints, and full generated manifests.

`splits/` contains lean exports of the completed seed-4001 assignments. They retain `image_id`, binary label, generator and family, duplicate group, protocol, fold, role, evaluation group, and seed. Join these files to a locally generated `data/manifests/master_manifest.csv` for source Parquet locations and image metadata, or to a condition's `feature_index.csv` for CLIP embeddings.

Run `python scripts/inspect_datasets.py` after obtaining the pinned Tiny-GenImage revision to create the full master manifest. Then run `python scripts/make_splits.py` to reproduce the full split manifests.
