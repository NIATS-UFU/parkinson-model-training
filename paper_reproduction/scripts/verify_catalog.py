"""V2 gate — the catalog and the shipped results describe the same 97 cells.

Recomputes the 97 cell_ids from `catalog.all_cells()` and checks they are exactly
the cell_ids present in `results/cells.parquet` — no missing, no extras. Because
cell_id is a hash of the spec, this proves the shipped predictions correspond to
the catalog's declared experiments and nothing has drifted.

Exit 0 = match. Exit 1 = mismatch (prints what differs).

Usage:  python scripts/verify_catalog.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from parkinson_spiral.catalog import EXPECTED_N_CELLS, all_cells  # noqa: E402


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    cells_path = repo / "results" / "cells.parquet"
    if not cells_path.is_file():
        print(f"ERROR: {cells_path} not found", file=sys.stderr)
        return 1

    catalog = all_cells()
    catalog_ids = {c.cell_id for c in catalog}
    result_ids = set(pd.read_parquet(cells_path)["cell_id"])

    ok = True
    if len(catalog) != EXPECTED_N_CELLS:
        print(f"FAIL: catalog has {len(catalog)} cells, expected {EXPECTED_N_CELLS}")
        ok = False
    missing = catalog_ids - result_ids
    extra = result_ids - catalog_ids
    if missing:
        print(f"FAIL: {len(missing)} catalog cells missing from results: {sorted(missing)[:5]}")
        ok = False
    if extra:
        print(f"FAIL: {len(extra)} result cells not in catalog: {sorted(extra)[:5]}")
        ok = False

    if ok:
        print(f"PASS: catalog and results agree on all {len(catalog_ids)} cell_ids.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
