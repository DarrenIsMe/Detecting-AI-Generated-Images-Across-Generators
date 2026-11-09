"""Lightweight Part 3 tests: no dataset, model weights or GPU needed."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from part3.perturb import Perturbation, downscale, jpeg_compress, parse_perturbations  # noqa: E402
from part3.sampling import sample_group  # noqa: E402


def noise_image(width: int = 96, height: int = 64, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8), "RGB")


def mean_abs_error(a: Image.Image, b: Image.Image) -> float:
    return float(np.abs(np.asarray(a, float) - np.asarray(b, float)).mean())


def test_jpeg_keeps_size_and_mode_and_is_deterministic() -> None:
    image = noise_image()
    out = jpeg_compress(image, 75)
    assert out.size == image.size and out.mode == "RGB"
    assert np.array_equal(np.asarray(out), np.asarray(jpeg_compress(image, 75)))


def test_lower_jpeg_quality_loses_more_detail() -> None:
    image = noise_image()
    assert mean_abs_error(image, jpeg_compress(image, 30)) > mean_abs_error(image, jpeg_compress(image, 95))


def test_downscale_rounds_each_side_and_never_upscales() -> None:
    image = noise_image(256, 200)
    assert downscale(image, 0.5).size == (128, 100)
    assert downscale(image, 0.25).size == (64, 50)
    assert downscale(image, 1.0).size == (256, 200)


def test_combined_perturbation_resizes_then_compresses() -> None:
    image = noise_image(200, 100)
    both = Perturbation("both", scale=0.5, jpeg=75).apply(image)
    assert both.size == (100, 50)
    assert np.array_equal(np.asarray(both), np.asarray(jpeg_compress(downscale(image, 0.5), 75)))


def test_clean_perturbation_is_identity() -> None:
    image = noise_image()
    assert np.array_equal(np.asarray(Perturbation("clean").apply(image)), np.asarray(image))


def test_parse_keeps_order_and_requires_clean() -> None:
    perts = parse_perturbations({"clean": {}, "jpeg_q50": {"jpeg": 50}, "half": {"scale": 0.5}})
    assert [p.name for p in perts] == ["clean", "jpeg_q50", "half"]
    with pytest.raises(ValueError):
        parse_perturbations({"jpeg_q50": {"jpeg": 50}})


@pytest.mark.parametrize("bad", [{"jpeg": 0}, {"jpeg": 101}, {"scale": 0}, {"scale": 1.5}, {"blur": 2}])
def test_parse_rejects_invalid_settings(bad: dict) -> None:
    with pytest.raises(ValueError):
        parse_perturbations({"clean": {}, "bad": bad})


def make_rows(n_per_class: int) -> list[dict]:
    return [{"image_id": f"{label}-{i:04d}", "label": label} for label in ("0", "1") for i in range(n_per_class)]


def test_sample_is_balanced_deterministic_and_order_independent() -> None:
    rows = make_rows(50)
    first = sample_group(rows, 10, 4001, "ADM")
    again = sample_group(list(reversed(rows)), 10, 4001, "ADM")
    assert [r["image_id"] for r in first] == [r["image_id"] for r in again]
    assert sum(r["label"] == "0" for r in first) == sum(r["label"] == "1" for r in first) == 10


def test_sample_depends_on_seed_and_group() -> None:
    rows = make_rows(50)
    base = {r["image_id"] for r in sample_group(rows, 10, 4001, "ADM")}
    assert base != {r["image_id"] for r in sample_group(rows, 10, 4002, "ADM")}
    assert base != {r["image_id"] for r in sample_group(rows, 10, 4001, "GLIDE")}


def test_sample_none_or_oversized_keeps_everything() -> None:
    rows = make_rows(5)
    assert len(sample_group(rows, None, 4001, "ADM")) == 10
    assert len(sample_group(rows, 100, 4001, "ADM")) == 10
