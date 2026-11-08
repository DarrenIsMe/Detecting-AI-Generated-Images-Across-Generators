"""Measure the modal JPEG quality/subsampling of the training reals.

Training reals = unique label-0 train rows of the strict split plus every logo fold. Their union's
mode is the matched_jpeg recipe (the top-level `quality`/`subsampling` keys). Profiles of the
strict-train and LOGO-train reals are recorded too, and must have the same mode, so one recipe
serves both protocols. Writes results/part2/jpeg_profile.json.
"""
from __future__ import annotations

import os
from pathlib import Path

from common import project_root_from_script, write_json
from part2.data import iter_image_bytes, load_manifest, load_split_rows, small_image_ids
from part2.formats import jpeg_properties, profile_from_properties

STRICT_SPLIT = "biggan_to_diffusion_strict.csv"
# Real-image sets profiled separately; "union" is the one the matched_jpeg recipe comes from.
PROFILE_SETS = ("strict_train", "logo_train", "union")


def train_real_ids_by_protocol(splits_dir: Path) -> dict[str, list[str]]:
    """Sorted unique label-0 train image_ids of the strict split, of all logo folds, and of both."""
    def train_reals(paths: list[Path]) -> set[str]:
        return {row["image_id"] for path in paths for row in load_split_rows(path)
                if row["label"] == "0" and row["split"] == "train"}

    strict = train_reals([splits_dir / STRICT_SPLIT])
    logo = train_reals(sorted(splits_dir.glob("logo_*.csv")))
    return {"strict_train": sorted(strict), "logo_train": sorted(logo), "union": sorted(strict | logo)}


def train_real_ids(splits_dir: Path) -> list[str]:
    """Sorted unique image_ids of label-0 / train rows across the strict and logo splits."""
    return train_real_ids_by_protocol(splits_dir)["union"]


def per_protocol_profiles(properties: dict[str, tuple[int, str] | None],
                          ids_by_set: dict[str, list[str]], small: set[str]) -> dict[str, dict]:
    """JPEG profile of each id set from per-image properties; ids in ``small`` are excluded and counted."""
    profiles = {}
    for name, ids in ids_by_set.items():
        profile = profile_from_properties(properties[i] for i in ids if i not in small)
        profile["n_excluded_small"] = sum(i in small for i in ids)
        profiles[name] = profile
    return profiles


def check_modes_agree(profiles: dict[str, dict]) -> None:
    """Raise ValueError unless every profile has the same modal quality and subsampling."""
    modes = {name: (p["quality"], p["subsampling"]) for name, p in profiles.items()}
    if len(set(modes.values())) > 1:
        raise ValueError(f"modal JPEG (quality, subsampling) differs between training-real sets: {modes}; "
                         "one matched_jpeg recipe would not represent both protocols")


def main(root: Path | None = None) -> dict:
    """Profile the training reals under ``root`` (default: this repo); returns the written JSON."""
    root = Path(root) if root is not None else project_root_from_script(__file__)
    ids_by_set = train_real_ids_by_protocol(root / "data" / "splits")
    manifest = {row["image_id"]: row for row in load_manifest(root)}
    missing = [i for i in ids_by_set["union"] if i not in manifest]
    if missing:
        raise SystemExit(f"{len(missing)} train reals missing from the manifest, e.g. {missing[:3]}")
    rows = [manifest[i] for i in ids_by_set["union"]]
    # Images under the global size floor are excluded from every condition.
    small = small_image_ids(rows)
    kept = [row for row in rows if row["image_id"] not in small]
    # Read every image once; each set's profile is a subset of these measurements.
    properties = {row["image_id"]: jpeg_properties(data) for row, data in iter_image_bytes(kept, root)}
    profiles = per_protocol_profiles(properties, ids_by_set, small)
    check_modes_agree(profiles)
    # Top-level keys are the union profile, as before: quality/subsampling are the matched_jpeg recipe.
    result = {**profiles["union"], "per_protocol": {name: profiles[name] for name in PROFILE_SETS}}
    out = root / "results" / "part2" / "jpeg_profile.json"
    # Write atomically: part2_extract.py may be reading the profile at the same time.
    tmp = out.with_name(out.name + ".tmp")
    write_json(tmp, result)
    os.replace(tmp, out)
    for name in PROFILE_SETS:
        p = profiles[name]
        print(f"{name}: n={p['n']} quality={p['quality']} subsampling={p['subsampling']}")
    return result


if __name__ == "__main__":
    main()
