"""Build the unified image manifest from datasets you have placed on disk.

Only needed for Layer-2 re-training (re-running experiments from scratch). Layer-1
table/figure regeneration does not need any images.

Expected layout under <data-root> (see docs/DATA.md):
    <data-root>/kaggle/{spiral,wave}/{training,testing}/{healthy,parkinson}/*.png
    <data-root>/folador/HOG-IA-desenho-manuscrito/pyTremorDetect/dataset-junto/...
    <data-root>/hemato/output/all_metadata.parquet   (Hemato is available on request)

Writes a parquet manifest with one row per image plus the assigned 5-fold
participant-grouped cv_fold. A dataset that is absent is simply skipped (its rows
are omitted), so you can, e.g., build a Kaggle+Folador-only manifest.

Usage:  python scripts/build_manifest.py --data-root <path> [--out results/manifest.parquet]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from parkinson_spiral.data import build_unified_manifest, assign_subject_folds  # noqa: E402


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True, help="directory holding kaggle/ folador/ hemato/")
    ap.add_argument("--out", type=Path, default=repo / "results" / "manifest.parquet")
    args = ap.parse_args()

    manifest = build_unified_manifest(args.data_root)
    if manifest.empty:
        raise SystemExit(
            f"no images found under {args.data_root}. Check the layout in docs/DATA.md."
        )
    manifest = assign_subject_folds(manifest)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(args.out, index=False)

    by_ds = manifest.groupby("dataset").size().to_dict()
    print(f"wrote {len(manifest)} rows -> {args.out}")
    print(f"datasets: {by_ds}")
    print(f"spiral rows: {int((manifest['drawing_type'] == 'spiral').sum())}")


if __name__ == "__main__":
    main()
