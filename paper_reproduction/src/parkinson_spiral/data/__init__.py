"""Data layer: unified manifest builders and subject-level splits."""

from .splits import (
    CV_DATASETS,
    assign_subject_folds,
    produce_subject_clean_resplit,
    published_test_mask,
    published_train_mask,
)
from .unified import (
    MANIFEST_COLUMNS,
    MANIFEST_DTYPES,
    VALID_DATASETS,
    build_unified_manifest,
    enforce_schema,
    make_image_id,
)

__all__ = [
    "MANIFEST_COLUMNS",
    "MANIFEST_DTYPES",
    "VALID_DATASETS",
    "build_unified_manifest",
    "enforce_schema",
    "make_image_id",
    "assign_subject_folds",
    "produce_subject_clean_resplit",
    "published_train_mask",
    "published_test_mask",
    "CV_DATASETS",
]
