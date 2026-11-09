"""Image perturbations for the robustness study: JPEG compression and downscaling.

Every perturbation is a pure function of a PIL image, so the same input always gives the same
output. That makes each condition reproducible and lets the unit tests check it without data.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image

ALLOWED_KEYS = frozenset({"jpeg", "scale"})


def jpeg_compress(image: Image.Image, quality: int) -> Image.Image:
    """Encode as JPEG at ``quality`` in memory and decode it again (RGB).

    Pillow's default chroma subsampling (4:2:0) is kept on purpose: it is what typical
    uploads and screenshots use, so this models real-world compression.
    """
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=int(quality))
    with Image.open(io.BytesIO(buffer.getvalue())) as decoded:
        return decoded.convert("RGB")


def downscale(image: Image.Image, scale: float) -> Image.Image:
    """Shrink both sides by ``scale`` (bicubic). The image is NOT upscaled back: the detector's
    own pipeline handles the size, exactly as it would for a small uploaded image."""
    if scale == 1.0:
        return image.convert("RGB")
    width, height = image.size
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.convert("RGB").resize(size, Image.Resampling.BICUBIC)


@dataclass(frozen=True)
class Perturbation:
    """One named test condition. ``scale`` is applied first, then ``jpeg``."""

    name: str
    scale: float | None = None
    jpeg: int | None = None

    def apply(self, image: Image.Image) -> Image.Image:
        out = image.convert("RGB")
        if self.scale is not None:
            out = downscale(out, self.scale)
        if self.jpeg is not None:
            out = jpeg_compress(out, self.jpeg)
        return out

    def spec(self) -> dict:
        """The settings recorded in output files and run metadata."""
        return {"name": self.name, "scale": self.scale, "jpeg": self.jpeg}


def parse_perturbations(block: dict) -> list[Perturbation]:
    """Build perturbations from the ``perturbations`` block of config/part3.yaml (order kept).

    Raises ValueError on unknown keys or out-of-range values, so a typo in the config cannot
    silently become an unperturbed run. Exactly one perturbation must be ``clean`` (no settings);
    every degradation is measured against it.
    """
    if not block:
        raise ValueError("config has no perturbations")
    out = []
    for name, settings in block.items():
        settings = settings or {}
        unknown = set(settings) - ALLOWED_KEYS
        if unknown:
            raise ValueError(f"perturbation {name!r}: unknown keys {sorted(unknown)}")
        scale, quality = settings.get("scale"), settings.get("jpeg")
        if scale is not None and not 0.0 < float(scale) <= 1.0:
            raise ValueError(f"perturbation {name!r}: scale must be in (0, 1], got {scale}")
        if quality is not None and not 1 <= int(quality) <= 100:
            raise ValueError(f"perturbation {name!r}: jpeg quality must be 1-100, got {quality}")
        out.append(Perturbation(
            name=str(name),
            scale=None if scale is None else float(scale),
            jpeg=None if quality is None else int(quality),
        ))
    clean = [p for p in out if p.scale is None and p.jpeg is None]
    if len(clean) != 1 or clean[0].name != "clean":
        raise ValueError("perturbations need exactly one unperturbed entry, named 'clean'")
    return out
