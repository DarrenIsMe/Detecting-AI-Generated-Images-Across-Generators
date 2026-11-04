from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from common import BKTree, UnionFind, resolve_configured_path, stable_id  # noqa: E402
from extract_clip import (  # noqa: E402
    LEGACY_CHECKPOINT_SHA256,
    LEGACY_PREPROCESS_SHA256,
    LEGACY_PROCESSING,
    can_upgrade_verified_legacy_cache,
    canonical_hash,
    input_fingerprint,
    load_image_batch,
)
from format_ablation import validate_required_rows  # noqa: E402
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


def test_all_failed_image_batch_advances_position(tmp_path: Path) -> None:
    rows = [
        {"image_id": "missing-1", "image_path": "missing-1.png"},
        {"image_id": "missing-2", "image_path": "missing-2.png"},
    ]
    next_start, tensors, usable_rows, failures = load_image_batch(
        rows, start=0, batch_size=2, root=tmp_path, preprocess=lambda image: image
    )
    assert next_start == len(rows)
    assert tensors == []
    assert usable_rows == []
    assert [item["image_id"] for item in failures] == ["missing-1", "missing-2"]


def test_cache_signature_changes_with_processing_configuration() -> None:
    baseline = {"processing": {"output_size": 224}, "checkpoint": "abc"}
    changed = {"processing": {"output_size": 256}, "checkpoint": "abc"}
    assert canonical_hash(baseline) != canonical_hash(changed)


def test_cache_signature_changes_with_checkpoint() -> None:
    baseline = {"processing": {"output_size": 224}, "checkpoint": "abc"}
    changed = {"processing": {"output_size": 224}, "checkpoint": "def"}
    assert canonical_hash(baseline) != canonical_hash(changed)


def test_input_fingerprint_changes_with_processed_bytes(tmp_path: Path) -> None:
    image_path = tmp_path / "image.bin"
    image_path.write_bytes(b"first")
    rows = [{"image_id": "image-1", "image_path": "image.bin"}]
    first = input_fingerprint(rows, tmp_path)
    image_path.write_bytes(b"second")
    assert input_fingerprint(rows, tmp_path) != first


def test_only_attested_legacy_cache_can_be_upgraded() -> None:
    metadata = {
        "model_name": "ViT-B-32-quickgelu",
        "pretrained": "openai",
        "variant": "standardized",
        "l2_normalize": True,
    }
    checkpoint = {"files": [{"sha256": LEGACY_CHECKPOINT_SHA256}]}
    attestation = {"input_fingerprint": "inputs", "feature_file_sha256": "features"}
    assert can_upgrade_verified_legacy_cache(
        metadata, "standardized", LEGACY_PROCESSING["standardized"],
        LEGACY_PREPROCESS_SHA256, checkpoint, "inputs", "features", attestation,
    )
    assert not can_upgrade_verified_legacy_cache(
        metadata, "standardized", LEGACY_PROCESSING["standardized"],
        LEGACY_PREPROCESS_SHA256, checkpoint, "changed", "features", attestation,
    )


def test_ablation_rejects_missing_required_features() -> None:
    rows = [
        {"image_id": "real", "label": "0", "split": "test"},
        {"image_id": "fake", "label": "1", "split": "test"},
    ]
    vectors = {"real": np.zeros(2, dtype=np.float32)}
    with pytest.raises(ValueError, match="missing 1 required images"):
        validate_required_rows(rows, vectors, "test.csv", ("test",))


def test_ablation_rejects_split_without_both_classes() -> None:
    rows = [{"image_id": "real", "label": "0", "split": "test"}]
    vectors = {"real": np.zeros(2, dtype=np.float32)}
    with pytest.raises(ValueError, match="requires both classes"):
        validate_required_rows(rows, vectors, "test.csv", ("test",))
