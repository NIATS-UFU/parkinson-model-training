"""Export the released model: a DINOv2 linear probe fit on 100% of Hemato.

This is the one trained artifact the paper releases (the H4 novelty: a frozen
public DINOv2 backbone + a tiny logistic-regression probe). The backbone is NOT
saved — it is downloaded from the public facebookresearch/dinov2 hub at use time.
Only the ~KB sklearn probe is ours; it is what this script fits and serialises.

This is CPU-runnable but MUST be run by you, on a machine that has the Hemato
dataset (it is IRB-restricted and not shipped). It:

  1. Builds a Hemato-only manifest + 5 participant-grouped folds.
  2. SANITY CHECK: reproduces the paper's pooled-OOF 5-fold Hemato ceiling
     (subject accuracy ~0.739) before doing the all-data fit. Aborts on mismatch.
  3. Fits the DINOv2 probe on ALL Hemato spiral images (AUG-None) — no holdout.
  4. Serialises the probe with skops + writes a config json + the model card + license.

Output (default models/exported/): parkinson_spiral_dinov2_hemato.skops,
config.json, README.md (the model card), LICENSE (OpenRAIL-M) — a folder ready to
`huggingface_hub.upload_folder`.

Usage:
  python models/export_hemato_probe.py --data-root <path-with-hemato/>
  python models/export_hemato_probe.py --manifest results/manifest.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from parkinson_spiral import DEFAULT_SEED  # noqa: E402
from parkinson_spiral.data import assign_subject_folds, build_unified_manifest  # noqa: E402
from parkinson_spiral.data.dataset import PreprocessedSpiralDataset  # noqa: E402
from parkinson_spiral.pipelines.dinov2_probe import DinoV2Probe  # noqa: E402
from parkinson_spiral.recipes import get_recipe  # noqa: E402

PAPER_HEMATO_CEILING = 0.739
SANITY_TOL = 0.02  # the fit is deterministic on CPU; allow a small margin for backbone/runtime drift.


def _hemato_spiral(manifest: pd.DataFrame) -> pd.DataFrame:
    m = manifest[(manifest["dataset"] == "hemato") & (manifest["drawing_type"] == "spiral")]
    if m.empty:
        raise SystemExit("no Hemato spiral rows found — is the Hemato dataset present under --data-root?")
    return m.reset_index(drop=True)


def _load_manifest(args) -> pd.DataFrame:
    if args.manifest:
        manifest = pd.read_parquet(args.manifest)
        if "cv_fold" not in manifest.columns or manifest["cv_fold"].isna().all():
            manifest = assign_subject_folds(manifest)
        return manifest
    manifest = build_unified_manifest(args.data_root)
    return assign_subject_folds(manifest)


def _oof_sanity(hem: pd.DataFrame) -> float:
    """Pooled out-of-fold subject accuracy across the 5 Hemato folds (paper ceiling)."""
    from parkinson_spiral.metrics import aggregate_to_subject

    recipe = get_recipe("AUG-None")
    oof = []
    for fold in range(5):
        tr = hem[hem["cv_fold"] != fold].reset_index(drop=True)
        te = hem[hem["cv_fold"] == fold].reset_index(drop=True)
        if tr.empty or te.empty:
            continue
        probe = DinoV2Probe()
        probe.fit(PreprocessedSpiralDataset(tr, recipe, is_train=True), seed=DEFAULT_SEED)
        proba = probe.predict_proba(PreprocessedSpiralDataset(te, recipe, is_train=False))
        oof.append(
            pd.DataFrame({
                "subject_id": te["subject_id"].to_numpy(),
                "y_true": (te["class_label"] == "parkinson").astype(int).to_numpy(),
                "y_proba_pos": proba[:, 1],
            })
        )
    pooled = pd.concat(oof, ignore_index=True)
    subj = aggregate_to_subject(pooled)
    return float((subj["y_true"].to_numpy() == (subj["y_proba_pos"].to_numpy() >= 0.5).astype(int)).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--data-root", type=Path, help="dir holding hemato/ (manifest built on the fly)")
    g.add_argument("--manifest", type=Path, help="prebuilt manifest parquet (scripts/build_manifest.py)")
    ap.add_argument("--out", type=Path, default=REPO / "models" / "exported")
    ap.add_argument("--skip-sanity", action="store_true", help="skip the 5-fold OOF reproduction check")
    args = ap.parse_args()

    try:
        import skops.io as sio
    except ImportError as err:
        raise SystemExit("skops is required: pip install -e '.[hf]'") from err

    hem = _hemato_spiral(_load_manifest(args))
    print(f"Hemato spiral images: {len(hem)}  subjects: {hem['subject_id'].nunique()}")

    if not args.skip_sanity:
        acc = _oof_sanity(hem)
        print(f"pooled-OOF 5-fold subject accuracy: {acc:.3f} (paper: {PAPER_HEMATO_CEILING})")
        if abs(acc - PAPER_HEMATO_CEILING) > SANITY_TOL:
            raise SystemExit(
                f"SANITY FAIL: OOF ceiling {acc:.3f} differs from paper {PAPER_HEMATO_CEILING} "
                f"by more than {SANITY_TOL}. Not exporting — investigate the data/backbone first."
            )
        print("sanity check PASS")

    # All-data fit (no holdout): this is the released probe.
    recipe = get_recipe("AUG-None")
    probe = DinoV2Probe()
    probe.fit(PreprocessedSpiralDataset(hem, recipe, is_train=True), seed=DEFAULT_SEED)

    args.out.mkdir(parents=True, exist_ok=True)
    model_path = args.out / "parkinson_spiral_dinov2_hemato.skops"
    sio.dump(probe._probe, model_path)

    config = {
        "backbone": "facebookresearch/dinov2 :: dinov2_vitb14 (public, frozen, NOT included)",
        "feature": "768-d CLS token",
        "probe": "sklearn LogisticRegression(C=1.0, max_iter=2000)",
        "recipe": "AUG-None (resize 224, ImageNet normalize, RGB)",
        "image_size": 224,
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
        "label_mapping": {"0": "healthy", "1": "parkinson"},
        "positive_class": "parkinson",
        "predict_proba_column": 1,
        "trained_on": "100% of Hemato spiral images (no holdout)",
        "n_train_images": int(len(hem)),
        "n_train_subjects": int(hem["subject_id"].nunique()),
        "seed": DEFAULT_SEED,
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    card_src = REPO / "models" / "MODEL_CARD.md"
    if card_src.is_file():
        (args.out / "README.md").write_text(card_src.read_text(encoding="utf-8"), encoding="utf-8")

    license_src = REPO / "models" / "MODEL_LICENSE.md"
    if license_src.is_file():
        (args.out / "LICENSE").write_text(license_src.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"wrote {model_path}")
    print(f"wrote {args.out / 'config.json'}")
    print(f"folder ready for Hugging Face: {args.out}")


if __name__ == "__main__":
    main()
