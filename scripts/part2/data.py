"""Manifest and Parquet byte access for the Part 2 detector."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq

from common import read_csv

MANIFEST_RELATIVE = Path("data") / "manifests" / "master_manifest.csv"


def load_manifest(root: Path) -> list[dict]:
    """Rows of the master manifest whose processing_status is "ok"."""
    rows = read_csv(Path(root) / MANIFEST_RELATIVE)
    return [row for row in rows if row.get("processing_status") == "ok"]


def small_image_ids(rows: list[dict], min_side: int = 112) -> set[str]:
    """image_ids whose shorter side is below min_side (from width/height columns)."""
    return {
        row["image_id"]
        for row in rows
        if min(int(row["width"]), int(row["height"])) < min_side
    }


def iter_image_bytes(rows: list[dict], root: Path) -> Iterator[tuple[dict, bytes]]:
    """Yield (row, encoded image bytes), shard by shard (sorted), reading each shard once.

    `shard` may be repo-relative or absolute; `root / shard` handles both because
    joining an absolute path discards the root.
    """
    by_shard: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_shard[row["shard"]].append(row)
    for shard in sorted(by_shard):
        column = pq.read_table(Path(root) / shard, columns=["image"]).column("image").to_pylist()
        # Within a shard, go in row order so output order is deterministic.
        for row in sorted(by_shard[shard], key=lambda r: int(r["row_in_shard"])):
            yield row, column[int(row["row_in_shard"])]["bytes"]


def load_split_rows(path: Path) -> list[dict]:
    """Rows of a split CSV (all values strings)."""
    return read_csv(Path(path))
