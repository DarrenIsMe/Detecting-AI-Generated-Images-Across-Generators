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


# ---- run script: LOGO folds share their real images (needs the Part 2 modules, so torch)


def test_logo_folds_share_reals_but_each_image_is_embedded_once() -> None:
    pytest.importorskip("torch")
    from part2.protocols import ProtocolSplit
    from part3_run import build_sample, effective_format

    reals = [{"image_id": f"r{i}", "label": "0", "generator": "real"} for i in range(4)]

    def fold(name: str) -> ProtocolSplit:
        fakes = [{"image_id": f"{name}{i}", "label": "1", "generator": name} for i in range(4)]
        return ProtocolSplit("logo", name, [], [], {name: reals + fakes})

    by_id = {r["image_id"]: {"label": r["label"], "shard": "s", "row_in_shard": 0}
             for r in reals + [{"image_id": f"{g}{i}", "label": "1"} for g in ("ADM", "SD15") for i in range(4)]}
    members, images = build_sample([fold("ADM"), fold("SD15")], by_id, None, 4001)
    assert len(members) == 16                    # 8 rows per fold
    assert len(images) == 12                     # 4 shared reals + 4 fakes per fold, each once
    assert len({m["evaluation_group"] for m in members}) == 2
    # Reals are never re-encoded, so matched_jpeg reuses their raw embedding.
    assert effective_format("matched_jpeg", 0) == "raw"
    assert effective_format("matched_jpeg", 1) == "matched_jpeg"


def test_threaded_embedding_matches_part2_embed_images(tmp_path: Path) -> None:
    """The threaded image preparation must give exactly what Part 2's embed_images gives,
    skip images too small for pipeline B, and not depend on the number of threads."""
    torch = pytest.importorskip("torch")
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    import io

    from part2.backbones import Backbone
    from part2.formats import JpegSpec, load_condition_image
    from part2.scoring import embed_images
    from part3_run import embed_condition, effective_format

    # Reals as JPEG, fakes as PNG; one 120 px image becomes too small for B at 0.75x.
    specs = [(0, "JPEG", (200, 160)), (1, "PNG", (256, 256)), (1, "PNG", (120, 120)), (0, "JPEG", (300, 180))]
    blobs = []
    for k, (_, fmt, size) in enumerate(specs):
        buffer = io.BytesIO()
        noise_image(*size, seed=k).save(buffer, fmt)
        blobs.append({"bytes": buffer.getvalue(), "path": f"img{k}"})
    pq.write_table(pa.table({"image": blobs}), tmp_path / "shard.parquet")
    images = [{"image_id": f"id{k}", "label": label, "shard": "shard.parquet", "row_in_shard": k}
              for k, (label, _, _) in enumerate(specs)]

    weights = torch.randn(3 * 4 * 4, 8, generator=torch.Generator().manual_seed(0))

    def encode(batch):  # deterministic stand-in for CLIP: pool, project, normalise
        f = torch.nn.functional.adaptive_avg_pool2d(batch, 4).reshape(len(batch), -1) @ weights
        return (f / f.norm(dim=1, keepdim=True)).numpy().astype(np.float32)

    backbone = Backbone("clip_l14", 8, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25), encode, "cpu", {})
    combos = [("A", "matched_jpeg"), ("A", "raw"), ("B", "matched_jpeg"), ("B", "raw")]
    jpeg = {"raw": None, "matched_jpeg": JpegSpec(96, "4:4:4")}
    pert = Perturbation("resize_0.75_jpeg_q50", scale=0.75, jpeg=50)

    one = embed_condition(pert, images, combos, jpeg, backbone, tmp_path, batch_size=2, workers=1)
    four = embed_condition(pert, images, combos, jpeg, backbone, tmp_path, batch_size=2, workers=4)
    for combo in combos:
        assert np.array_equal(one[combo], four[combo], equal_nan=True)

    for k, (label, _, _) in enumerate(specs):
        for pipe, fmt in combos:
            eff = effective_format(fmt, label)
            image = pert.apply(load_condition_image(blobs[k]["bytes"], label, eff, jpeg.get(eff)))
            if pipe == "B" and min(image.size) < 112:
                assert np.isnan(one[(pipe, fmt)][k]).all()  # skipped, not crashed
            else:
                expected = embed_images([image], backbone, pipe)[0]
                np.testing.assert_allclose(one[(pipe, fmt)][k], expected, rtol=1e-6, atol=1e-6)


# ---- analysis script: counting flips and confusion cells


def tiny_predictions():
    pd = pytest.importorskip("pandas")
    rows = []
    # (image, label, p clean, p after jpeg): f1 evades, f2 stays caught, r1 becomes a false alarm.
    for image, label, clean, after in (("f1", 1, 0.9, 0.2), ("f2", 1, 0.8, 0.7), ("r1", 0, 0.1, 0.6), ("r2", 0, 0.2, 0.1)):
        for pert, p in (("clean", clean), ("jpeg_q50", after)):
            rows.append({"head": "h", "pipeline": "A", "format": "raw", "perturbation": pert, "image_id": image,
                         "evaluation_group": "ADM", "in_overall": True, "label": label, "generator": "g",
                         "scored": True, "p_fake": p})
    return pd.DataFrame(rows)


def test_flip_table_counts_evasions_and_new_false_alarms() -> None:
    pytest.importorskip("torch")
    from part3_analysis import flip_table

    overall = flip_table(tiny_predictions(), ["clean", "jpeg_q50"]).set_index("evaluation_group").loc["overall"]
    assert overall["fake_caught_clean"] == 2 and overall["fake_missed_after"] == 1
    assert overall["real_flagged_after"] == 1 and overall["real_cleared_after"] == 0
    assert overall["mean_p_fake_change_fakes"] == pytest.approx(((0.2 - 0.9) + (0.7 - 0.8)) / 2)


def test_confusion_counts_use_the_half_threshold() -> None:
    pytest.importorskip("torch")
    from part3_analysis import confusion_counts

    after = confusion_counts(tiny_predictions()).set_index("perturbation").loc["jpeg_q50"]
    assert (after["true_fake_pred_fake"], after["true_fake_pred_real"]) == (1, 1)
    assert (after["true_real_pred_fake"], after["true_real_pred_real"]) == (1, 1)
