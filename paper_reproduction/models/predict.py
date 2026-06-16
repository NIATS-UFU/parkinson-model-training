"""Run the released DINOv2+Hemato probe on a folder of spiral images -> P(PD).

Loads the public DINOv2 backbone (downloaded from torch.hub, never shipped), the
AUG-None preprocessing, and our serialised logistic probe, then prints the
probability of Parkinson's for each image. CPU is fine.

Usage:
  python models/predict.py --model-dir models/exported --images path/to/spirals/
  python models/predict.py --model-dir models/exported --images one_spiral.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from parkinson_spiral.recipes import get_recipe  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _iter_images(images: Path):
    if images.is_file():
        yield images
        return
    for p in sorted(images.iterdir()):
        if p.suffix.lower() in IMAGE_EXTS:
            yield p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, default=REPO / "models" / "exported")
    ap.add_argument("--images", type=Path, required=True, help="a spiral image or a folder of them")
    args = ap.parse_args()

    try:
        import skops.io as sio
        import torch
    except ImportError as err:
        raise SystemExit("torch and skops are required: pip install -e '.[train,hf]'") from err

    model_path = args.model_dir / "parkinson_spiral_dinov2_hemato.skops"
    if not model_path.is_file():
        raise SystemExit(f"probe not found at {model_path}. Run models/export_hemato_probe.py first.")
    config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))

    # Load the trusted probe. skops requires opting into the (sklearn) types it contains.
    untrusted = sio.get_untrusted_types(file=model_path)
    probe = sio.load(model_path, trusted=untrusted)

    # Public frozen backbone (downloaded once, cached by torch.hub).
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    backbone.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = backbone.to(device)

    transform = get_recipe("AUG-None", image_size=config.get("image_size", 224)).eval_transform
    pd_col = int(config.get("predict_proba_column", 1))

    paths = list(_iter_images(args.images))
    if not paths:
        raise SystemExit(f"no images found at {args.images}")

    print(f"{'image':50s}  P(parkinson)")
    with torch.inference_mode():
        for p in paths:
            img = Image.open(p).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            feat = backbone(tensor).detach().cpu().numpy()
            proba = probe.predict_proba(feat)[0, pd_col]
            print(f"{p.name:50s}  {float(proba):.3f}")


if __name__ == "__main__":
    main()
