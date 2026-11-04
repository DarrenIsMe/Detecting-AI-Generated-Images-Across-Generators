# Generated manifests

Full manifests are not committed because each is approximately 18–22 MB and they can be reproduced from the pinned dataset revision.

- `master_manifest.csv`: audit metadata, hashes, encodings, dimensions, labels, generators, and duplicate groups.
- `<variant>_manifest.csv`: processed image path and dimensions for one preprocessing condition.

Generate them with `inspect_datasets.py` and `prepare_images.py`. The `.gitignore` prevents accidental publication of generated manifest CSVs containing local absolute paths.
