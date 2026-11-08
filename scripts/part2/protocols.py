"""Train/validation/test protocols (strict, logo, final) with exclusion and leakage checks."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from part2.data import load_split_rows


@dataclass
class ProtocolSplit:
    """One train/validation/test arrangement; rows are manifest-style dicts."""

    protocol: str
    fold: str
    train: list[dict]
    validation: list[dict]
    test_groups: dict[str, list[dict]]
    separate_groups: dict[str, list[dict]] = field(default_factory=dict)


def _partition(rows: list[dict], exclude: frozenset[str]) -> tuple[list[dict], list[dict], dict[str, list[dict]]]:
    """Split rows into train, validation and per-evaluation_group test lists, dropping excluded ids."""
    train: list[dict] = []
    validation: list[dict] = []
    groups: dict[str, list[dict]] = {}
    for row in rows:
        if row["image_id"] in exclude:
            continue
        if row["split"] == "train":
            train.append(row)
        elif row["split"] == "validation":
            validation.append(row)
        elif row["split"] == "test":
            groups.setdefault(row["evaluation_group"], []).append(row)
    return train, validation, groups


def _logo_splits(splits_dir: Path, exclude: frozenset[str]) -> list[ProtocolSplit]:
    """Seven leave-one-generator-out folds, sorted by held-out generator."""
    folds = []
    for path in sorted(splits_dir.glob("logo_*.csv")):
        rows = load_split_rows(path)
        held_out = rows[0]["fold"]
        train, validation, groups = _partition(rows, exclude)
        # The single test group is named after the held-out generator.
        folds.append(ProtocolSplit("logo", held_out, train, validation, {held_out: groups.get(held_out, [])}))
    return sorted(folds, key=lambda s: s.fold)


def _dedupe(rows: list[dict]) -> list[dict]:
    """Keep the first row per image_id, preserving order."""
    seen: set[str] = set()
    out = []
    for row in rows:
        if row["image_id"] not in seen:
            seen.add(row["image_id"])
            out.append(row)
    return out


def load_protocol(protocol: str, splits_dir: Path, exclude_ids: frozenset[str] = frozenset()) -> list[ProtocolSplit]:
    """Build the splits of a protocol; ``exclude_ids`` are removed from every list."""
    splits_dir = Path(splits_dir)
    if protocol == "strict":
        train, validation, groups = _partition(load_split_rows(splits_dir / "biggan_to_diffusion_strict.csv"), exclude_ids)
        _, _, separate = _partition(load_split_rows(splits_dir / "biggan_to_midjourney_separate_test.csv"), exclude_ids)
        return [ProtocolSplit("strict", "strict", train, validation, groups, separate)]
    if protocol == "logo":
        return _logo_splits(splits_dir, exclude_ids)
    if protocol == "final":
        folds = _logo_splits(splits_dir, exclude_ids)
        train = _dedupe([r for f in folds for r in f.train])
        validation = _dedupe([r for f in folds for r in f.validation])
        # A duplicate group must not straddle train and validation: drop it from both.
        shared = {r["duplicate_group_id"] for r in train} & {r["duplicate_group_id"] for r in validation}
        train = [r for r in train if r["duplicate_group_id"] not in shared]
        validation = [r for r in validation if r["duplicate_group_id"] not in shared]
        return [ProtocolSplit("final", "all", train, validation, {})]
    raise ValueError(f"unknown protocol: {protocol!r}")


def assert_no_leakage(split: ProtocolSplit) -> None:
    """Raise ValueError on overlap between train/validation/test or protocol-rule violations."""
    test = [r for g in (*split.test_groups.values(), *split.separate_groups.values()) for r in g]
    parts = {"train": split.train, "validation": split.validation, "test": test}
    names = list(parts)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            for key in ("image_id", "duplicate_group_id"):
                shared = {r[key] for r in parts[names[a]]} & {r[key] for r in parts[names[b]]}
                if shared:
                    raise ValueError(f"{key} shared by {names[a]} and {names[b]}: {sorted(shared)[:3]}")
    if split.protocol == "logo" and any(r["generator"] == split.fold for r in split.train + split.validation):
        raise ValueError(f"held-out generator {split.fold} appears in train/validation")
    if split.protocol == "strict":
        bad = {r["generator"] for r in split.train + split.validation if r["label"] == "1"} - {"BigGAN"}
        if bad:
            raise ValueError(f"strict train/validation contains non-BigGAN fakes: {sorted(bad)}")
