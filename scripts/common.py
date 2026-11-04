from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import yaml


GENERATOR_NAMES = {
    0: "Real",
    1: "ADM",
    2: "BigGAN",
    3: "GLIDE",
    4: "Midjourney",
    5: "SD14",
    6: "SD15",
    7: "VQDM",
    8: "Wukong",
}

GENERATOR_FAMILIES = {
    "Real": "real",
    "ADM": "diffusion",
    "BigGAN": "gan",
    "GLIDE": "diffusion",
    "Midjourney": "unknown",
    "SD14": "diffusion",
    "SD15": "diffusion",
    "VQDM": "diffusion",
    "Wukong": "diffusion",
}


def project_root_from_script(script_file: str) -> Path:
    return Path(script_file).resolve().parents[1]


def load_config(project_root: Path, config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else project_root / "config" / "datasets.yaml"
    if not path.is_absolute():
        path = project_root / path
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_configured_path(
    project_root: Path,
    configured_path: str | Path,
    environment_variable: str | None = None,
) -> Path:
    """Resolve an optional environment override, then an absolute or project-relative path."""
    override = os.environ.get(environment_variable, "") if environment_variable else ""
    path = Path(override or configured_path).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def stable_id(*parts: object, length: int = 24) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def deterministic_shuffle(values: list[Any], seed: int, salt: str = "") -> list[Any]:
    result = list(values)
    rng = random.Random(stable_id(seed, salt, length=16))
    rng.shuffle(result)
    return result


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


class BKTree:
    def __init__(self) -> None:
        self.root: tuple[int, int, dict[int, Any]] | None = None

    @staticmethod
    def distance(left: int, right: int) -> int:
        return (left ^ right).bit_count()

    def add(self, value: int, index: int) -> None:
        if self.root is None:
            self.root = (value, index, {})
            return
        node = self.root
        while True:
            distance = self.distance(value, node[0])
            child = node[2].get(distance)
            if child is None:
                node[2][distance] = (value, index, {})
                return
            node = child

    def query(self, value: int, radius: int) -> list[tuple[int, int]]:
        if self.root is None:
            return []
        matches: list[tuple[int, int]] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = self.distance(value, node[0])
            if distance <= radius:
                matches.append((node[1], distance))
            low, high = distance - radius, distance + radius
            stack.extend(child for edge, child in node[2].items() if low <= edge <= high)
        return matches
