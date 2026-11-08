"""Frozen feature backbones (torchvision ResNet-50, OpenCLIP ViT-B/32 and ViT-L/14)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from common import load_config, resolve_configured_path

# ImageNet statistics used by the torchvision ResNet-50 weights.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalisation(kind: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """(mean, std) input normalisation of a backbone kind, available without loading any model.

    load_backbone uses this too, so the normalisation a head records is the one its features used.
    """
    if kind == "open_clip":
        # OpenAI CLIP statistics; importing open_clip.constants loads no weights.
        from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

        return tuple(OPENAI_DATASET_MEAN), tuple(OPENAI_DATASET_STD)
    if kind == "torchvision":
        return IMAGENET_MEAN, IMAGENET_STD
    raise ValueError(f"unsupported backbone kind {kind!r}")


@dataclass
class Backbone:
    name: str
    dim: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    # Normalised CPU batch (N,3,224,224) -> L2-normalised float32 (N, dim).
    encode: Callable[[torch.Tensor], np.ndarray]
    device: str
    # The cfg["backbones"][name] block it was built from (None for hand-made backbones);
    # score_images compares it with the block recorded in a head.
    config: dict | None = None


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    """Row-wise L2 normalisation; zero rows stay finite via the norm clamp."""
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _make_encode(forward: Callable[[torch.Tensor], torch.Tensor], device: str):
    """Wrap a model forward into the fp32, inference-mode, L2-normalised numpy encoder."""

    def encode(batch: torch.Tensor) -> np.ndarray:
        with torch.inference_mode():
            features = forward(batch.to(device=device, dtype=torch.float32))
            return l2_normalize(features.float()).cpu().numpy().astype(np.float32)

    return encode


def _freeze(model: torch.nn.Module, device: str) -> torch.nn.Module:
    """fp32, eval mode, no gradients, on the target device."""
    model = model.float().to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_backbone(name: str, cfg: dict, root: Path, device: str = "auto") -> Backbone:
    """Build the frozen backbone `name` from the `backbones` block of the Part 2 config."""
    spec = cfg["backbones"][name]  # KeyError for unknown names
    mean, std = normalisation(spec["kind"])  # ValueError for unsupported kinds
    device = _resolve_device(device)

    if spec["kind"] == "open_clip":
        import open_clip

        # Checkpoint cache dir comes from Part 1's dataset config (env override supported).
        clip_cfg = load_config(root).get("clip", {})
        cache_dir = resolve_configured_path(
            root, clip_cfg.get("cache_dir", "data/models/open_clip"), "SC4001_CLIP_CACHE_DIR"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        model, _, _ = open_clip.create_model_and_transforms(
            spec["model_name"], pretrained=spec["pretrained"], cache_dir=str(cache_dir)
        )
        model = _freeze(model, device)
        return Backbone(
            name, int(spec["dim"]), mean, std,
            _make_encode(model.encode_image, device), device, dict(spec),
        )

    # torchvision (normalisation() above has already rejected any other kind)
    import torchvision

    weights = torchvision.models.ResNet50_Weights[spec["weights"]]
    model = torchvision.models.resnet50(weights=weights)
    model.fc = torch.nn.Identity()  # expose the 2048-d pooled features
    model = _freeze(model, device)
    return Backbone(
        name, int(spec["dim"]), mean, std,
        _make_encode(model, device), device, dict(spec),
    )
