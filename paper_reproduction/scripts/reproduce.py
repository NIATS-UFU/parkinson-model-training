"""Layer 2 — re-run the experiments from scratch over the 97-cell catalog.

This trains every model and reproduces the predictions in `results/`. It is
idempotent: a cell whose predictions already exist is skipped, so an interrupted
run resumes where it left off, and pointing it at the shipped `results/` is a
no-op. Needs a built manifest (scripts/build_manifest.py) and the datasets on disk;
some pipelines need a GPU (see docs/REPRODUCE.md).

Cells are keyed by cell_id, so re-running into a *fresh* output directory and then
comparing with `scripts/verify_catalog.py` confirms the catalog matches.

Examples:
  python scripts/reproduce.py --dry-run                 # list the 97 cells, no training
  python scripts/reproduce.py --manifest results/manifest.parquet --out runs/fresh
  python scripts/reproduce.py --manifest ... --pipeline dinov2_probe   # CPU subset only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from parkinson_spiral.catalog import all_cells  # noqa: E402
from parkinson_spiral.runner import RunPaths, run_cell  # noqa: E402

# Which pipelines need a GPU in practice (documented in docs/REPRODUCE.md).
GPU_PIPELINES = {"huang_vgg19", "dogan_triple"}


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=repo / "results" / "manifest.parquet")
    ap.add_argument("--out", type=Path, default=repo / "runs" / "reproduce")
    ap.add_argument("--pipeline", action="append", default=None, help="restrict to these pipeline name(s); repeatable")
    ap.add_argument("--dry-run", action="store_true", help="list cells and exit (no training)")
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    args = ap.parse_args()

    cells = all_cells()
    if args.pipeline:
        keep = set(args.pipeline)
        cells = [c for c in cells if c.pipeline_name in keep]

    print(f"{len(cells)} cells selected")
    if args.dry_run:
        for c in cells:
            gpu = " [GPU]" if c.pipeline_name in GPU_PIPELINES else ""
            fold = f" fold={c.cv_fold}" if c.cv_fold is not None else ""
            print(
                f"  {c.cell_id}  {c.pipeline_name:13s} {c.recipe_name:9s} "
                f"{c.train_source}->{c.test_source} [{c.split_source}]{fold}{gpu}"
            )
        return

    if not args.manifest.is_file():
        raise SystemExit(
            f"manifest not found: {args.manifest}. Build one with scripts/build_manifest.py "
            "(Layer-2 re-training needs the datasets on disk; see docs/REPRODUCE.md)."
        )

    paths = RunPaths(manifest=args.manifest, outputs_dir=args.out)
    for i, spec in enumerate(cells, 1):
        row = run_cell(spec, paths, force=args.force)
        acc = row.get("subject_accuracy")
        acc_s = f"{acc:.3f}" if isinstance(acc, (int, float)) else "n/a"
        print(
            f"[{i}/{len(cells)}] {spec.pipeline_name} {spec.recipe_name} "
            f"{spec.train_source}->{spec.test_source} -> subj_acc={acc_s}"
        )

    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
