"""Reproducibility signature and cache contracts."""

from pathlib import Path

import pandas as pd
from parkinson_spiral.runner import RunPaths, _is_cached, _reproducibility_signature, contiguous_shard


def test_reproducibility_signature_is_canonical() -> None:
    """Equivalent payload mappings must produce the same signature."""
    assert _reproducibility_signature({"seed": 1, "fold": 2}) == _reproducibility_signature({"fold": 2, "seed": 1})


def test_cache_requires_matching_run_signature(tmp_path: Path) -> None:
    """Reject unsigned and differently signed cached cell outputs."""
    paths = RunPaths(manifest=tmp_path / "manifest.parquet", outputs_dir=tmp_path / "results")
    paths.predictions_dir.mkdir(parents=True)
    (paths.predictions_dir / "cell.parquet").touch()

    pd.DataFrame([{"cell_id": "cell"}]).to_parquet(paths.cells_path)
    assert _is_cached(paths, "cell", "expected") is False

    pd.DataFrame([{"cell_id": "cell", "cache_signature": "different"}]).to_parquet(paths.cells_path)
    assert _is_cached(paths, "cell", "expected") is False

    pd.DataFrame([{"cell_id": "cell", "cache_signature": "expected"}]).to_parquet(paths.cells_path)
    assert _is_cached(paths, "cell", "expected") is True


def test_contiguous_shards_are_balanced_disjoint_and_complete() -> None:
    """Partition the catalog without duplication while preserving local order."""
    items = list(range(11))
    shards = [contiguous_shard(items, shard_count=2, shard_index=index) for index in range(2)]

    assert shards == [list(range(5)), list(range(5, 11))]
    assert shards[0] + shards[1] == items
