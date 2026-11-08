"""Preprocessing pipelines A and B for the Part 2 detector.

Pipeline A: OpenCLIP/OpenAI geometry (bicubic resize of the short side to 224, centre crop).
Pipeline B: a 112x112 native-pixel crop, centred and snapped down to a 16-pixel grid
(so JPEG block alignment is preserved), then bicubic upscaled to 224x224.

The geometry is hard-coded below. config/part2.yaml repeats it (and min_side) so that it is
recorded in feature specs and head metadata; check_config_matches_code() makes the CLIs and the
scoring API fail loudly if the two ever disagree.
"""
from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

OUTPUT_SIZE = 224
# Pipeline B: native crop size and the grid its top-left corner snaps down to.
B_CROP = 112
B_GRID = 16
# Shorter-side floor of Part 2's exclusion rule: the smallest image pipeline B can crop.
# It is also the "min_side<112" reason extract_features records for skipped images.
MIN_SIDE = B_CROP

# Pipeline A transforms are stateless, so build them once.
_PIPELINE_A = T.Compose(
    [T.Resize(OUTPUT_SIZE, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(OUTPUT_SIZE)]
)


class ImageTooSmall(ValueError):
    """Raised when the shorter image side is smaller than the pipeline-B crop."""


def check_config_matches_code(cfg: dict) -> None:
    """Raise ValueError if config/part2.yaml's pipeline geometry or min_side differ from this module.

    The config values are what feature specs and head metadata record, but the constants above
    are what actually runs, so a silent mismatch would label features with settings they were not
    made with. Called at start-up by part2_extract.py and part2_run.py, and by score_images.
    """
    expected = {"A": {"size": OUTPUT_SIZE}, "B": {"crop": B_CROP, "grid": B_GRID, "size": OUTPUT_SIZE}}
    pipelines = cfg.get("pipelines") or {}
    problems = [
        f"pipelines.{name}: config {pipelines.get(name)!r}, code {expected.get(name)!r}"
        for name in sorted(set(expected) | set(pipelines))
        if pipelines.get(name) != expected.get(name)
    ]
    if cfg.get("min_side") != MIN_SIDE:
        problems.append(f"min_side: config {cfg.get('min_side')!r}, code {MIN_SIDE} "
                        f"(the pipeline B crop and the 'min_side<{MIN_SIDE}' skip reason)")
    if problems:
        raise ValueError(
            "config/part2.yaml disagrees with the preprocessing hard-coded in scripts/part2/preprocess.py: "
            + "; ".join(problems)
            + ". Change the code and the config together, then re-extract with part2_extract.py --overwrite."
        )


def pipeline_b_box(width: int, height: int, crop: int = B_CROP, grid: int = B_GRID) -> tuple[int, int, int, int]:
    """Return the (left, top, right, bottom) native crop box for pipeline B."""
    if min(width, height) < crop:
        raise ImageTooSmall(f"image {width}x{height} is smaller than the {crop}px crop")
    # Centre the crop, then snap the top-left corner down onto the grid.
    left = ((width - crop) // 2) // grid * grid
    top = ((height - crop) // 2) // grid * grid
    return left, top, left + crop, top + crop


def apply_pipeline(image: Image.Image, pipeline: str) -> Image.Image:
    """Apply pipeline "A" or "B" and return a 224x224 RGB image."""
    if pipeline not in ("A", "B"):
        raise ValueError(f"unknown pipeline {pipeline!r}; expected 'A' or 'B'")
    image = image.convert("RGB")
    if pipeline == "A":
        return _PIPELINE_A(image)
    # Crop native pixels first, only then upscale.
    box = pipeline_b_box(*image.size)
    return image.crop(box).resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.Resampling.BICUBIC)


def to_tensor(
    image: Image.Image, mean: tuple[float, float, float], std: tuple[float, float, float]
) -> torch.Tensor:
    """Convert a 224x224 RGB image to a normalised 3x224x224 float32 tensor."""
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    return ((tensor - mean_t) / std_t).contiguous()
