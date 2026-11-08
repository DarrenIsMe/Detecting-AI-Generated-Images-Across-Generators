"""File-format helpers: JPEG profile estimation and the matched-JPEG condition."""
from __future__ import annotations

import io
import json
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from PIL import Image, JpegImagePlugin

SUBSAMPLING_BY_CODE = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}
VALID_SUBSAMPLING = frozenset(SUBSAMPLING_BY_CODE.values())
# Histogram bucket for every other layout: grayscale, CMYK, 4:4:0, 4:1:1, ... (never a valid recipe).
OTHER_SUBSAMPLING = "gray/other"


@dataclass(frozen=True)
class JpegSpec:
    """A JPEG encoding recipe: quality 1-100 and chroma subsampling string."""

    quality: int
    subsampling: str


def decode_rgb(data: bytes) -> Image.Image:
    """Decode encoded bytes to a fully loaded RGB image (no EXIF transpose)."""
    with Image.open(io.BytesIO(data)) as image:
        return image.convert("RGB")


@lru_cache(maxsize=1)
def _reference_tables() -> dict[int, tuple[int, ...]]:
    """Quantisation table 0 of a blank 8x8 JPEG saved by Pillow at each quality 1-100."""
    tables = {}
    for quality in range(1, 101):
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buffer, format="JPEG", quality=quality)
        with Image.open(io.BytesIO(buffer.getvalue())) as ref:
            tables[quality] = tuple(ref.quantization[0])
    return tables


def estimate_jpeg_quality(image: Image.Image) -> int:
    """Pillow quality whose reference table is closest (L1) to the image's table 0."""
    quantization = getattr(image, "quantization", None)
    if not quantization:
        raise ValueError("image has no JPEG quantisation tables")
    table = tuple(quantization[0])
    # min() keeps the first (lowest) quality on exact ties.
    return min(
        _reference_tables().items(),
        key=lambda item: sum(abs(a - b) for a, b in zip(item[1], table)),
    )[0]


def jpeg_subsampling(image: Image.Image) -> str:
    """Chroma subsampling of a JPEG as "4:4:4", "4:2:2", "4:2:0", else "gray/other"."""
    return SUBSAMPLING_BY_CODE.get(JpegImagePlugin.get_sampling(image), OTHER_SUBSAMPLING)


def load_condition_image(data: bytes, label: int, fmt: str, jpeg: JpegSpec | None) -> Image.Image:
    """Decode to RGB; for matched_jpeg, re-encode fakes (label 1) with the real-image JPEG recipe."""
    if fmt == "matched_jpeg" and jpeg is None:
        raise ValueError("matched_jpeg requires a JpegSpec")
    image = decode_rgb(data)
    if fmt == "matched_jpeg" and label == 1:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg.quality, subsampling=jpeg.subsampling)
        image = decode_rgb(buffer.getvalue())
    return image


def jpeg_properties(data: bytes) -> tuple[int, str] | None:
    """(estimated quality, subsampling) of encoded bytes, or None when they are not a JPEG.

    Only the headers are read (Image.open is lazy), so measuring is cheap.
    """
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "JPEG":
            return None
        return estimate_jpeg_quality(image), jpeg_subsampling(image)


def measure_jpeg_profile(samples: Iterable[bytes]) -> dict:
    """Modal JPEG quality and subsampling over encoded samples, with histograms.

    Non-JPEG samples (no quantisation tables) are counted in `n_non_jpeg` and not in `n`.
    """
    return profile_from_properties(jpeg_properties(data) for data in samples)


def profile_from_properties(properties: Iterable[tuple[int, str] | None]) -> dict:
    """The profile of measure_jpeg_profile from per-image jpeg_properties (None = non-JPEG).

    Lets one pass over the images serve several subsets (e.g. per-protocol training reals).
    """
    qualities: Counter = Counter()
    subsamplings: Counter = Counter()
    n_non_jpeg = 0
    for prop in properties:
        if prop is None:
            n_non_jpeg += 1
            continue
        quality, subsampling = prop
        qualities[quality] += 1
        subsamplings[subsampling] += 1
    n = sum(qualities.values())
    if n == 0:
        raise ValueError("no JPEG samples to measure")
    # Mode with a deterministic tie-break (higher quality / lexicographically larger name).
    return {
        "quality": max(qualities, key=lambda q: (qualities[q], q)),
        "subsampling": max(subsamplings, key=lambda s: (subsamplings[s], s)),
        "n": n,
        "n_non_jpeg": n_non_jpeg,
        "quality_hist": {str(q): c for q, c in sorted(qualities.items())},
        "subsampling_hist": dict(sorted(subsamplings.items())),
    }


def load_jpeg_spec(path: Path) -> JpegSpec:
    """Read the modal recipe from a jpeg_profile.json."""
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    spec = JpegSpec(int(profile["quality"]), str(profile["subsampling"]))
    if spec.subsampling not in VALID_SUBSAMPLING:
        raise ValueError(f"unsupported subsampling in profile: {spec.subsampling}")
    return spec
