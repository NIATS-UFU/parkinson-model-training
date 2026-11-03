"""Tests for the embedding export's manifest-correspondence checks.

`build_embedding_frame` is the pure half of `export-embeddings`, so both ways the
join can silently go wrong are exercised here without downloading a backbone.
"""

import numpy as np
import pandas as pd
import pytest


def _manifest(labels: list[str]) -> pd.DataFrame:
    n = len(labels)
    return pd.DataFrame({
        "image_id": [f"kaggle:{i:04d}" for i in range(n)],
        "dataset": ["kaggle"] * n,
        "subject_id": [f"kaggle:V{i:02d}" for i in range(n)],
        "session_id": [None] * n,
        "drawing_type": ["spiral"] * n,
        "class_label": labels,
        "cv_fold": [0] * n,
    })


def test_one_row_per_image_with_features_appended(paper) -> None:
    """Every manifest image gets exactly one row carrying its full feature vector."""
    manifest = _manifest(["healthy", "parkinson", "parkinson"])
    features = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
    labels = np.array([0, 1, 1])

    frame = paper.build_embedding_frame(manifest, features, labels)

    assert len(frame) == 3
    assert list(frame.columns) == [*paper.EMBEDDING_METADATA_COLUMNS, "f000", "f001", "f002", "f003"]
    assert frame["image_id"].tolist() == manifest["image_id"].tolist()
    np.testing.assert_array_equal(frame[["f000", "f001", "f002", "f003"]].to_numpy(), features)


def test_row_count_mismatch_is_rejected(paper) -> None:
    """Fewer vectors than images means the extractor dropped rows."""
    manifest = _manifest(["healthy", "parkinson"])
    features = np.zeros((1, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="returned 1 vectors for 2 images"):
        paper.build_embedding_frame(manifest, features, np.array([0]))


def test_reordered_labels_are_rejected(paper) -> None:
    """Labels out of manifest order mean the join would attach the wrong subject.

    This is the failure that would not announce itself: the frame would have the
    right shape and every downstream distance would be computed on mislabelled
    points.
    """
    manifest = _manifest(["healthy", "parkinson"])
    features = np.zeros((2, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="do not follow the manifest order"):
        paper.build_embedding_frame(manifest, features, np.array([1, 0]))


def test_fallback_label_map_matches_the_canonical_one(paper, monkeypatch) -> None:
    """The duplicated literal must never drift from the mapping it stands in for.

    `class_label_to_int` keeps a copy of `CLASS_LABEL_TO_INT` so the Layer-1 path
    works without torch. A copy that silently disagreed with the canonical mapping
    would mislabel every exported embedding while looking perfectly healthy, so
    this forces the fallback branch and compares it against the real thing.

    Skipped where torch is absent: there is no canonical mapping to compare to.
    """
    import sys

    canonical = pytest.importorskip(
        "parkinson_spiral.data.dataset",
        reason="torch is in the [train] extra; nothing to compare the fallback against",
    ).CLASS_LABEL_TO_INT

    # A None entry in sys.modules makes `import` raise ImportError, which is the
    # branch a torch-free install takes.
    monkeypatch.setitem(sys.modules, "parkinson_spiral.data.dataset", None)
    assert paper.class_label_to_int() == canonical
