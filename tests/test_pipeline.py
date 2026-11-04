from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from common import BKTree, UnionFind, resolve_configured_path, stable_id  # noqa: E402
from prepare_images import standardized_image  # noqa: E402


def test_stable_id_is_deterministic() -> None:
    assert stable_id("a", 1) == stable_id("a", 1)
    assert stable_id("a", 1) != stable_id("a", 2)


def test_standardization_center_crops_to_requested_size() -> None:
    image = Image.new("RGB", (640, 320), "red")
    output = standardized_image(image, 224)
    assert output.size == (224, 224)
    assert output.mode == "RGB"


def test_union_find_groups_members() -> None:
    union = UnionFind(4)
    union.union(0, 1)
    union.union(2, 3)
    assert union.find(0) == union.find(1)
    assert union.find(0) != union.find(2)


def test_bk_tree_finds_near_hash() -> None:
    tree = BKTree()
    tree.add(0b0000, 0)
    tree.add(0b1111, 1)
    assert tree.query(0b0001, 1) == [(0, 1)]


def test_configured_path_environment_override(monkeypatch, tmp_path: Path) -> None:
    external = tmp_path / "external-data"
    monkeypatch.setenv("SC4001_TEST_DATA", str(external))
    assert resolve_configured_path(tmp_path, "data/default", "SC4001_TEST_DATA") == external.resolve()
