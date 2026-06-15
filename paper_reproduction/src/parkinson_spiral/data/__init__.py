"""Data layer: unified manifest builders, the torch Dataset, and subject-level splits."""
from .unified import (
    MANIFEST_COLUMNS,
    MANIFEST_DTYPES,
    VALID_DATASETS,
    build_unified_manifest,
    enforce_schema,
    make_image_id,
)
from .dataset import PreprocessedSpiralDataset, CLASS_LABEL_TO_INT
from .splits import (
    assign_subject_folds,
    produce_subject_clean_resplit,
    published_train_mask,
    published_test_mask,
    CV_DATASETS,
)

__all__ = [
    "MANIFEST_COLUMNS",
    "MANIFEST_DTYPES",
    "VALID_DATASETS",
    "build_unified_manifest",
    "enforce_schema",
    "make_image_id",
    "PreprocessedSpiralDataset",
    "CLASS_LABEL_TO_INT",
    "assign_subject_folds",
    "produce_subject_clean_resplit",
    "published_train_mask",
    "published_test_mask",
    "CV_DATASETS",
]
