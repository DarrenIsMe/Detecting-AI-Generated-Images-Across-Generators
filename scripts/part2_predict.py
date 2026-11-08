"""Score one image with a saved Part 2 head and print a single JSON line."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Silence a harmless Hugging Face symlink warning on Windows.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from PIL import Image

from part2.scoring import score_images


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    args = parser.parse_args(argv)

    with Image.open(args.image) as img:
        img.load()
        p_fake = float(score_images([img], args.head)[0])
    label = "AI-generated" if p_fake >= 0.5 else "real"
    print(json.dumps({"p_fake": p_fake, "label": label}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
