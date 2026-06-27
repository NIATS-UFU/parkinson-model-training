"""The catalog enumerates exactly the 97 paper cells, all valid and unique."""

from parkinson_spiral.catalog import (
    EXPECTED_N_CELLS,
    all_cells,
    group_a,
    group_b,
    group_c,
)


def test_total_is_97():
    assert EXPECTED_N_CELLS == 97
    assert len(all_cells()) == 97


def test_group_sizes_match_paper():
    assert len(group_a()) == 70  # in-domain CV ceilings
    assert len(group_b()) == 24  # cross-dataset off-diagonals
    assert len(group_c()) == 3  # published-split fidelity


def test_cell_ids_unique():
    ids = [c.cell_id for c in all_cells()]
    assert len(set(ids)) == len(ids)


def test_every_cell_validates():
    for c in all_cells():
        c.validate()


def test_folador_reports_svm_linear_head():
    folador = [c for c in all_cells() if c.pipeline_name == "folador_hog"]
    assert folador
    assert all(c.pipeline_kwargs.get("classifier") == "SVM_linear" for c in folador)
