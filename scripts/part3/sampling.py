"""Seeded, class-balanced sampling of strict test rows."""
from __future__ import annotations

import random


def sample_group(rows: list[dict], per_class: int | None, seed: int, group: str) -> list[dict]:
    """Pick ``per_class`` real and ``per_class`` fake rows from one evaluation group.

    Rows are sorted by image_id before drawing, so the sample depends only on the split file,
    the seed and the group name, never on file or dictionary order. ``per_class=None`` keeps
    every row; a class with fewer rows than ``per_class`` is kept whole. Output is sorted by
    image_id.
    """
    if per_class is None:
        return sorted(rows, key=lambda r: r["image_id"])
    rng = random.Random(f"{seed}:{group}")  # one stream per group: adding a group changes no other
    picked: list[dict] = []
    for label in ("0", "1"):
        members = sorted((r for r in rows if r["label"] == label), key=lambda r: r["image_id"])
        picked.extend(members if len(members) <= per_class else rng.sample(members, per_class))
    return sorted(picked, key=lambda r: r["image_id"])
