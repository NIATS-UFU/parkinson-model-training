"""The 97 experiment cells behind the paper, in the paper's naming.

This is the declarative source of truth for what was run. `scripts/paper.py verify-catalog`
checks that the cell_ids enumerated here exactly match the cell_ids in a local
`results/cells.parquet` (no missing, no extras), and `scripts/paper.py reproduce` re-runs
these specs to regenerate the predictions from scratch.

Grouping mirrors the paper:
  Group A (70)  in-domain 5-fold CV ceilings  -> Table "cv-ceiling", matrix diagonals
  Group B (24)  cross-dataset off-diagonals   -> Table "cross-dataset", matrix off-diagonals, asymmetry
  Group C (3)   in-domain published-split fidelity -> Table "b1-fidelity" Released-split column
"""

from . import DEFAULT_SEED
from .protocol import CellSpec

SOURCES = ("kaggle", "folador", "hemato")
N_FOLDS = 5

# Folador HOG reports its SVM_linear head in the paper.
_FOLADOR_KW = {"classifier": "SVM_linear"}
# The bounded M4 reproduction uses all three public backbones as frozen feature
# extractors. Literal three-backbone fine-tuning is not feasible because the
# TensorFlow runtime exposes no Apple Metal device.
_DOGAN_KW = {"fine_tune": False}


def _cv(pipeline: str, recipe: str, source: str, pipeline_kwargs: dict | None = None) -> list[CellSpec]:
    """The 5 within-dataset CV-fold cells for one (pipeline, recipe, dataset) diagonal."""
    return [
        CellSpec(
            pipeline_name=pipeline,
            recipe_name=recipe,
            train_source=source,
            test_source=source,
            split_source="cv-fold",
            cv_fold=fold,
            task_filter="spiral",
            target_column="class_label",
            augment_then_split=False,
            seed=DEFAULT_SEED,
            pipeline_kwargs=dict(pipeline_kwargs or {}),
            recipe_kwargs={},
        )
        for fold in range(N_FOLDS)
    ]


def _cross(pipeline: str, recipe: str, pipeline_kwargs: dict | None = None) -> list[CellSpec]:
    """The 6 off-diagonal cross-dataset transfer cells for one (pipeline, recipe)."""
    return [
        CellSpec(
            pipeline_name=pipeline,
            recipe_name=recipe,
            train_source=train,
            test_source=test,
            split_source="cross-dataset",
            cv_fold=None,
            task_filter="spiral",
            target_column="class_label",
            augment_then_split=False,
            seed=DEFAULT_SEED,
            pipeline_kwargs=dict(pipeline_kwargs or {}),
            recipe_kwargs={},
        )
        for train in SOURCES
        for test in SOURCES
        if train != test
    ]


def _published(
    pipeline: str, recipe: str, source: str, augment_then_split: bool, pipeline_kwargs: dict | None = None
) -> CellSpec:
    return CellSpec(
        pipeline_name=pipeline,
        recipe_name=recipe,
        train_source=source,
        test_source=source,
        split_source="published",
        cv_fold=None,
        task_filter="spiral",
        target_column="class_label",
        augment_then_split=augment_then_split,
        seed=DEFAULT_SEED,
        pipeline_kwargs=dict(pipeline_kwargs or {}),
        recipe_kwargs={},
    )


def group_a() -> list[CellSpec]:
    """In-domain CV ceilings (70 fold-cells)."""
    cells: list[CellSpec] = []
    # DINOv2 frozen probe — AUG-None across all three datasets (matrix diagonals).
    for source in SOURCES:
        cells += _cv("dinov2_probe", "AUG-None", source)
    # DINOv2 aug-robustness on Hemato (H4): the same frozen probe under Huang/Dogan aug.
    cells += _cv("dinov2_probe", "AUG-Huang", "hemato")
    cells += _cv("dinov2_probe", "AUG-Dogan", "hemato")
    # Literature pipelines on their own recipe across all three datasets.
    for source in SOURCES:
        cells += _cv("huang_vgg19", "AUG-Huang", source)
    for source in SOURCES:
        cells += _cv("dogan_triple", "AUG-Dogan", source, _DOGAN_KW)
    for source in SOURCES:
        cells += _cv("folador_hog", "AUG-None", source, _FOLADOR_KW)
    return cells


def group_b() -> list[CellSpec]:
    """Cross-dataset off-diagonals (24 cells)."""
    return (
        _cross("dinov2_probe", "AUG-None")
        + _cross("huang_vgg19", "AUG-Huang")
        + _cross("dogan_triple", "AUG-Dogan", _DOGAN_KW)
        + _cross("folador_hog", "AUG-None", _FOLADOR_KW)
    )


def group_c() -> list[CellSpec]:
    """In-domain published-split fidelity (3 cells)."""
    return [
        # Huang VGG19 on Kaggle's released split.
        _published("huang_vgg19", "AUG-Huang", "kaggle", augment_then_split=False),
        # Dogan triple on Kaggle's released split. The source paper reports augmentation
        # before splitting, but does not publish enough split detail to reproduce that
        # leakage exactly; this release therefore keeps the auditable released split and
        # applies augmentation only to its training partition.
        _published(
            "dogan_triple",
            "AUG-Dogan",
            "kaggle",
            augment_then_split=False,
            pipeline_kwargs=_DOGAN_KW,
        ),
        # Folador HOG+SVM on the Folador dataset's released split.
        _published("folador_hog", "AUG-None", "folador", augment_then_split=False, pipeline_kwargs=_FOLADOR_KW),
    ]


def all_cells() -> list[CellSpec]:
    """All 97 paper cells."""
    cells = group_a() + group_b() + group_c()
    ids = [c.cell_id for c in cells]
    if len(set(ids)) != len(ids):
        raise AssertionError("catalog contains duplicate cell_ids")
    return cells


# Convenience for quick sanity checks: `python -c "from parkinson_spiral.catalog import all_cells; print(len(all_cells()))"`
EXPECTED_N_CELLS = 97
