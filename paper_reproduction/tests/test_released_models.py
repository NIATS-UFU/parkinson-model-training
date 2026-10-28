"""Tests for saving, reloading and verifying the released models.

The scikit-learn heads are fitted on small synthetic data, so the round trip runs without the
datasets, a GPU or the deep-learning backbones. The checks guard what would make a released model
silently differ from the one behind the paper: a file that reloads to a different state, a file
changed after saving, an overwritten model, and a comparison that hides a changed decision.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from parkinson_spiral.catalog import all_cells
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.svm import SVC


def _xy() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(40, 12))
    return features, (features[:, 0] > 0).astype(int)


def _cell(paper, name: str):
    return next(cell for cell in all_cells() if paper.released_model_name(cell) == name)


def _fitted_folador(paper):
    from parkinson_spiral.pipelines.folador_hog import FoladorHOG

    features, labels = _xy()
    pipeline = FoladorHOG(classifier="SVM_linear")
    pipeline._model = SVC(kernel="linear", random_state=0).fit(features, labels)
    return pipeline, features


def test_catalog_releases_one_model_per_pipeline_and_training_dataset(paper) -> None:
    """Twelve models, each the fit behind the two off-diagonal cells of its training dataset."""
    served: dict[str, list] = {}
    for cell in all_cells():
        name = paper.released_model_name(cell)
        if name is not None:
            served.setdefault(name, []).append(cell)

    assert len(served) == 12
    for cells in served.values():
        assert len(cells) == 2
        assert {cell.train_source for cell in cells} == {cells[0].train_source}
        assert cells[0].train_source not in {cell.test_source for cell in cells}


def test_cells_that_hold_data_out_release_no_model(paper) -> None:
    held_out = [cell for cell in all_cells() if cell.split_source in {"cv-fold", "published"}]

    assert held_out
    assert all(paper.released_model_name(cell) is None for cell in held_out)


def test_saved_model_reloads_and_predicts_the_same_decisions(paper, tmp_path: Path) -> None:
    pytest.importorskip("skops")
    pipeline, features = _fitted_folador(paper)
    spec = _cell(paper, "folador_hog-kaggle")

    directory = paper.save_released_model(spec, pipeline, tmp_path)
    reloaded, card = paper.load_released_model(directory)

    assert card["cells"] == {
        cell.cell_id: cell.test_source
        for cell in all_cells()
        if paper.released_model_name(cell) == "folador_hog-kaggle"
    }
    assert card["weight_vector_heads"] == ["classifier.skops"]
    np.testing.assert_allclose(
        reloaded._model.decision_function(features), pipeline._model.decision_function(features), rtol=0, atol=1e-12
    )


def test_two_part_model_with_a_random_state_round_trips(paper, tmp_path: Path) -> None:
    """Dogan's head is an NCA, which keeps a RandomState, and an SVM saved as its weight vector."""
    pytest.importorskip("skops")
    from parkinson_spiral.pipelines.dogan_triple import DoganTriple

    features, labels = _xy()
    pipeline = DoganTriple()
    pipeline._nca = NeighborhoodComponentsAnalysis(n_components=3, random_state=0).fit(features, labels)
    pipeline._svm = SVC(kernel="linear", random_state=0).fit(pipeline._nca.transform(features), labels)

    directory = paper.save_released_model(_cell(paper, "dogan_triple-hemato"), pipeline, tmp_path)

    assert paper.same_fitted_state(pipeline._nca, paper._load_artifact(directory / "nca.skops"))
    assert paper.same_fitted_state(paper.linear_svm_head(pipeline._svm), paper._load_artifact(directory / "svm.skops"))


def test_saved_linear_svm_keeps_no_training_rows(paper, tmp_path: Path) -> None:
    """The released file holds the weight vector and intercept, not the support vectors."""
    pytest.importorskip("skops")
    pipeline, features = _fitted_folador(paper)

    directory = paper.save_released_model(_cell(paper, "folador_hog-kaggle"), pipeline, tmp_path)
    saved = paper._load_artifact(directory / "classifier.skops")

    assert not hasattr(saved, "support_vectors_")
    assert all(np.ndim(value) == 0 or np.shape(value)[0] != len(features) for value in vars(saved).values())


def test_linear_svm_head_refuses_a_nonlinear_kernel(paper) -> None:
    features, labels = _xy()
    rbf = SVC(kernel="rbf", random_state=0).fit(features, labels)

    with pytest.raises(ValueError, match="only a fitted linear-kernel SVC"):
        paper.linear_svm_head(rbf)


def test_torch_weights_round_trip_as_a_state_dict(paper, tmp_path: Path) -> None:
    """Huang's fine-tuned network is saved as tensors only and reloads to identical weights."""
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    network = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2))
    path = tmp_path / "vgg19_state_dict.pt"

    paper._dump_artifact(network, path)
    reloaded = paper._load_artifact(path)

    assert paper.same_fitted_state(network, reloaded)
    with torch.no_grad():
        network[0].weight[0, 0] += 1e-6
    assert paper.same_fitted_state(network, reloaded) is False


def test_fitted_state_comparison_detects_a_changed_weight(paper) -> None:
    pipeline, _ = _fitted_folador(paper)
    features, labels = _xy()
    changed = SVC(kernel="linear", random_state=0).fit(features, labels)
    changed.dual_coef_ = changed.dual_coef_ * 1.000001

    assert paper.same_fitted_state(pipeline._model, changed) is False


def test_a_file_changed_after_saving_is_refused(paper, tmp_path: Path) -> None:
    pytest.importorskip("skops")
    pipeline, _ = _fitted_folador(paper)
    directory = paper.save_released_model(_cell(paper, "folador_hog-kaggle"), pipeline, tmp_path)
    with (directory / "classifier.skops").open("ab") as file:
        file.write(b"\0")

    with pytest.raises(ValueError, match="does not match the checksum"):
        paper.load_released_model(directory)


def test_saving_refuses_to_overwrite(paper, tmp_path: Path) -> None:
    pytest.importorskip("skops")
    pipeline, _ = _fitted_folador(paper)
    spec = _cell(paper, "folador_hog-kaggle")
    paper.save_released_model(spec, pipeline, tmp_path)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        paper.save_released_model(spec, pipeline, tmp_path)


def test_saving_refuses_a_cell_that_holds_data_out(paper, tmp_path: Path) -> None:
    fold_cell = next(cell for cell in all_cells() if cell.split_source == "cv-fold")

    with pytest.raises(ValueError, match="not an off-diagonal cell"):
        paper.save_released_model(fold_cell, object(), tmp_path)


@pytest.mark.parametrize(
    ("name", "kwargs", "message"),
    [
        ("dogan_triple-kaggle", {"fine_tune": True}, "fine-tuned backbones are not saved"),
        ("folador_hog-kaggle", {"classifier": "CNN1D"}, "CNN1D classifier is not supported"),
    ],
)
def test_configurations_whose_state_is_not_saved_are_refused(paper, name: str, kwargs: dict, message: str) -> None:
    from dataclasses import replace

    spec = replace(_cell(paper, name), pipeline_kwargs=kwargs)

    with pytest.raises(ValueError, match=message):
        paper.check_model_saveable(spec)


def test_model_saver_fails_before_training_when_models_exist(paper, tmp_path: Path) -> None:
    cells = [cell for cell in all_cells() if paper.released_model_name(cell) == "folador_hog-kaggle"]
    (tmp_path / "folador_hog-kaggle").mkdir()

    with pytest.raises(SystemExit, match="refusing to overwrite"):
        paper._model_saver(cells, tmp_path)


def test_model_saver_needs_an_off_diagonal_cell(paper, tmp_path: Path) -> None:
    fold_cells = [cell for cell in all_cells() if cell.split_source == "cv-fold"][:3]

    with pytest.raises(SystemExit, match="no selected cell trains a released"):
        paper._model_saver(fold_cells, tmp_path)


def _predictions(probabilities: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "image_id": [f"img{index}" for index in range(6)],
        "subject_id": ["s1", "s1", "s2", "s2", "s3", "s3"],
        "session_id": pd.array([None] * 6, dtype="string"),
        "y_true": [1, 1, 0, 0, 1, 1],
        "y_proba_pos": probabilities,
    })


def test_comparison_accepts_a_small_shift_that_changes_no_decision(paper) -> None:
    reference = _predictions([0.9, 0.8, 0.2, 0.3, 0.4, 0.45])
    shifted = _predictions([0.9, 0.8, 0.2, 0.3, 0.4, 0.4500001])

    result = paper.compare_cell_predictions(reference, shifted)

    assert result["same_test_images"] is True
    assert result["same_subject_metrics"] is True
    assert result["max_abs_proba_delta"] == pytest.approx(1e-7)


def test_comparison_flags_a_changed_participant_decision(paper) -> None:
    reference = _predictions([0.9, 0.8, 0.2, 0.3, 0.4, 0.45])
    flipped = _predictions([0.9, 0.8, 0.2, 0.3, 0.6, 0.65])

    assert paper.compare_cell_predictions(reference, flipped)["same_subject_metrics"] is False


def test_comparison_flags_different_test_images(paper) -> None:
    reference = _predictions([0.9, 0.8, 0.2, 0.3, 0.4, 0.45])
    other = reference.assign(image_id=[f"other{index}" for index in range(6)])

    result = paper.compare_cell_predictions(reference, other)

    assert result["same_test_images"] is False
    assert result["same_subject_metrics"] is False
