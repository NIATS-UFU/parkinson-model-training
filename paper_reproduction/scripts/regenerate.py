"""Layer 1 — regenerate the paper's tables and figures from the shipped results.

This is the headline reproducibility claim: no datasets, no GPU, no training. It
reads only `results/` (cells.parquet + predictions/) and rebuilds every table and
figure in seconds. Run this first.

Usage:  python scripts/regenerate.py [--out paper_artifacts]
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    # Delegate to the table/figure generator (single implementation, no duplication).
    script = Path(__file__).resolve().parent / "make_paper_tables_figures.py"
    sys.argv = [str(script), *sys.argv[1:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
