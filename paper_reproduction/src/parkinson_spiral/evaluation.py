"""Statistics reported with the benchmark: participant-level intervals and paired comparisons.

Reads a results directory (cells.parquet + predictions/) and never trains. Every number is at the
participant level: a participant's probability is the mean over their drawings, called
Parkinson's at 0.5 or above (`metrics.aggregate_to_subject`).

- Intervals. For each pipeline (native recipe) and each train/test pair, accuracy, sensitivity,
  specificity and ROC-AUC with a 95% percentile bootstrap over participants. The four metrics of
  one cell share one set of 1,000 resamples, drawn from the run's seed. A diagonal pools the
  out-of-fold predictions of its five folds.
- Paired comparisons. On each off-diagonal cell, every pipeline against the DINOv2 reference on
  the same participants: an exact McNemar test on the discordant participants, Holm's correction
  within each test set, and the Quesenberry-Hurst interval for the difference in error rates
  (Berrar and Lozano 2013, "Significance tests or confidence intervals: which are preferable for
  the comparison of classifiers?", J. Exp. Theor. Artif. Intell. 25(2)), at both the unadjusted
  and the Holm-adjusted level.
- Baselines. The majority-class accuracy of each test set.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.metrics import roc_auc_score

from .metrics import aggregate_to_subject, binary_metrics, percentile_bootstrap_ci

REFERENCE = "dinov2_probe"
ALPHA = 0.05
NATIVE_RECIPES = {
    "dinov2_probe": "AUG-None",
    "folador_hog": "AUG-None",
    "huang_vgg19": "AUG-Huang",
    "dogan_triple": "AUG-Dogan",
}
DATASETS = ("kaggle", "folador", "hemato")
METRICS = ("accuracy", "sensitivity", "specificity", "auc")


# Participant-level metrics at the 0.5 threshold. On a sample holding both classes each equals
# `binary_metrics`; a resample holding one class is counted by `cell_intervals`.
def _accuracy(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    return float((y_true == (y_proba >= 0.5).astype(int)).mean())


def _sensitivity(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    positive = y_true == 1
    return float(((y_proba >= 0.5) & positive).sum() / positive.sum()) if positive.any() else 0.0


def _specificity(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    negative = y_true == 0
    return float(((y_proba < 0.5) & negative).sum() / negative.sum()) if negative.any() else 0.0


def _auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) == 2 else float("nan")


METRIC_FUNCTIONS = {"accuracy": _accuracy, "sensitivity": _sensitivity, "specificity": _specificity, "auc": _auc}


def load_cells(results: Path) -> tuple[pd.DataFrame, int]:
    """Return (cells, seed) of a results directory; the run must use one seed."""
    cells = pd.read_parquet(Path(results) / "cells.parquet")
    seeds = cells["seed"].unique()
    if len(seeds) != 1:
        raise ValueError(f"{results}: cells.parquet holds {len(seeds)} seeds; expected one")
    return cells, int(seeds[0])


def native_cells(cells: pd.DataFrame) -> list[dict]:
    """The 12 diagonal and 24 off-diagonal cells of the native recipes, in a fixed order.

    A diagonal lists its five fold cells in cells.parquet order, the order they are pooled in.
    """
    out = []
    for pipeline in sorted(NATIVE_RECIPES):
        recipe = NATIVE_RECIPES[pipeline]
        for train in DATASETS:
            for test in DATASETS:
                diagonal = train == test
                split = "cv-fold" if diagonal else "cross-dataset"
                hit = cells[
                    (cells.pipeline_name == pipeline)
                    & (cells.recipe_name == recipe)
                    & (cells.train_source == train)
                    & (cells.test_source == test)
                    & (cells.split_source == split)
                ]
                expected = 5 if diagonal else 1
                if len(hit) != expected:
                    raise ValueError(f"{pipeline} {train}->{test}: {len(hit)} {split} cells, expected {expected}")
                out.append({
                    "pipeline": pipeline,
                    "recipe": recipe,
                    "train": train,
                    "test": test,
                    "kind": "diagonal" if diagonal else "off-diagonal",
                    "cell_ids": list(hit.cell_id),
                })
    return out


def subject_frame(results: Path, cell_ids: list[str]) -> pd.DataFrame:
    """Participant-level frame of one cell, or of the pooled folds of a diagonal."""
    frames = [pd.read_parquet(Path(results) / "predictions" / f"{cell_id}.parquet") for cell_id in cell_ids]
    return aggregate_to_subject(pd.concat(frames, ignore_index=True))


def cell_intervals(subjects: pd.DataFrame, seed: int) -> dict:
    """Point value and 95% percentile interval of the four metrics on one shared resample set.

    `percentile_bootstrap_ci` recreates its generator from `seed` on every call, so all four
    metrics see the same resamples. Resamples holding one class only are counted: on them
    sensitivity or specificity is undefined, and a non-zero count means the interval needs care.
    """
    row: dict = {"n_subjects": len(subjects), "n_pd": int(subjects.y_true.sum())}
    one_class: list[bool] = []

    def count_one_class(y_true: np.ndarray, _y_proba: np.ndarray) -> float:
        one_class.append(len(np.unique(y_true)) < 2)
        return 0.0

    percentile_bootstrap_ci(subjects, count_one_class, seed=seed)
    row["n_one_class_resamples"] = int(sum(one_class[1:]))  # the first call scores the full sample
    for metric in METRICS:
        interval = percentile_bootstrap_ci(subjects, METRIC_FUNCTIONS[metric], seed=seed)
        row[metric] = interval["point"]
        row[f"{metric}_lo"] = interval["ci_lo"]
        row[f"{metric}_hi"] = interval["ci_hi"]
    reported = binary_metrics(subjects["y_true"].to_numpy(), subjects["y_proba_pos"].to_numpy())
    row["points_equal_binary_metrics"] = all(
        (np.isnan(row[metric]) and np.isnan(reported[metric])) or row[metric] == reported[metric] for metric in METRICS
    )
    return row


def intervals(results: Path) -> pd.DataFrame:
    """Intervals for the 36 native-recipe cells of a results directory."""
    cells, seed = load_cells(results)
    rows = []
    for cell in native_cells(cells):
        subjects = subject_frame(results, cell["cell_ids"])
        rows.append({**cell, "cell_ids": ";".join(cell["cell_ids"]), **cell_intervals(subjects, seed)})
    return pd.DataFrame(rows)


def baselines(results: Path) -> pd.DataFrame:
    """Majority-class accuracy of each test set, from one off-diagonal cell's participants."""
    cells, _ = load_cells(results)
    rows = []
    for test in DATASETS:
        cell = cells[(cells.test_source == test) & (cells.split_source == "cross-dataset")].iloc[0]
        subjects = subject_frame(results, [cell.cell_id])
        n_pd = int(subjects.y_true.sum())
        rows.append({
            "test": test,
            "n_subjects": len(subjects),
            "n_pd": n_pd,
            "n_control": len(subjects) - n_pd,
            "majority_baseline": max(n_pd, len(subjects) - n_pd) / len(subjects),
        })
    return pd.DataFrame(rows)


def qh_interval(b: int, c: int, n: int, alpha: float) -> dict:
    """Quesenberry-Hurst interval for the difference between two classifiers' error rates.

    `b` counts participants the reference gets right and the competitor wrong, `c` the reverse,
    `n` all participants. Solving n(|p_b - p_c| - d)^2 / ((p_b + p_c) - d^2) <= chi2(1 - alpha, 1)
    for d (Berrar and Lozano 2013, eq. 4 and 5) gives d = [nD +/- sqrt(k(S(n + k) - nD^2))] / (n + k)
    with D = |p_b - p_c|, S = p_b + p_c and k the critical value. The bounds are computed on
    |p_b - p_c| and given its sign, so a positive interval means the reference errs less.
    """
    k = float(chi2.ppf(1.0 - alpha, 1))
    p_b, p_c = b / n, c / n
    signed = p_b - p_c
    distance = abs(signed)
    total = p_b + p_c
    centre = n * distance / (n + k)
    half = float(np.sqrt(max(k * (total * (n + k) - n * distance * distance), 0.0))) / (n + k)
    lo, hi = centre - half, centre + half
    if signed < 0:
        lo, hi = -hi, -lo
    return {
        "qh_point_shrunk": float(centre if signed >= 0 else -centre),
        "qh_lo": float(lo),
        "qh_hi": float(hi),
        "qh_crit_chi2": k,
        "qh_alpha": float(alpha),
    }


def mcnemar_exact(reference_correct: np.ndarray, competitor_correct: np.ndarray) -> dict:
    """Exact McNemar test: a two-sided binomial test on the discordant participants.

    With no discordant participant the two classifiers decided identically, so p = 1.
    """
    b = int(np.sum((reference_correct == 1) & (competitor_correct == 0)))
    c = int(np.sum((reference_correct == 0) & (competitor_correct == 1)))
    m = b + c
    p = 1.0 if m == 0 else float(binomtest(b, m, 0.5, alternative="two-sided").pvalue)
    return {"discordant_b": b, "discordant_c": c, "n_discordant": m, "p_mcnemar_exact": p}


def holm_levels(pvalues: list[float], alpha: float = ALPHA) -> tuple[list[float], list[float]]:
    """Holm's step-down procedure. Returns (adjusted p, Holm level), each in input order.

    The smallest p is judged at alpha/m, the next at alpha/(m - 1), and so on; the same level sets
    the width of the adjusted interval.
    """
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: np.inf if np.isnan(pvalues[i]) else pvalues[i])
    adjusted = [np.nan] * m
    levels = [np.nan] * m
    running = 0.0
    for rank, i in enumerate(order):
        levels[i] = alpha / (m - rank)
        if np.isnan(pvalues[i]):
            continue
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted, levels


def subject_table(predictions_dir: Path, cell_id: str) -> pd.DataFrame:
    """One row per participant of a cell: label, mean probability, decision and whether it is right."""
    predictions = pd.read_parquet(Path(predictions_dir) / f"{cell_id}.parquet")
    if not predictions.groupby("subject_id")["y_true"].nunique().eq(1).all():
        raise ValueError(f"cell {cell_id}: a participant carries more than one label")
    table = predictions.groupby("subject_id", sort=True).agg(
        y_true=("y_true", "first"),
        y_proba=("y_proba_pos", "mean"),
        n_images=("y_proba_pos", "size"),
    )
    table["y_pred"] = (table["y_proba"] >= 0.5).astype(int)
    table["correct"] = (table["y_pred"] == table["y_true"]).astype(int)
    return table.reset_index()


def paired(results: Path) -> pd.DataFrame:
    """Every pipeline against the reference on each off-diagonal cell: 18 accuracy comparisons.

    Holm's correction runs over the six comparisons that share a test set.
    """
    cells = pd.read_parquet(Path(results) / "cells.parquet")
    off = cells[(cells.train_source != cells.test_source) & (cells.cv_fold.isna())]
    if len(off) != 24:
        raise ValueError(f"expected 24 off-diagonal cells, found {len(off)}")
    pipelines = sorted(off.pipeline_name.unique())
    competitors = [pipeline for pipeline in pipelines if pipeline != REFERENCE]
    predictions_dir = Path(results) / "predictions"
    tables = {
        (row.train_source, row.test_source, row.pipeline_name): subject_table(predictions_dir, row.cell_id)
        for row in off.itertuples()
    }

    rows = []
    for test in sorted(off.test_source.unique()):
        trains = sorted(off[off.test_source == test].train_source.unique())
        participants = tables[(trains[0], test, REFERENCE)]["subject_id"].to_numpy()
        for train in trains:
            for pipeline in pipelines:
                if not np.array_equal(tables[(train, test, pipeline)]["subject_id"].to_numpy(), participants):
                    raise ValueError(f"{train}->{test} {pipeline}: participants differ from the reference's")
        n = len(participants)
        block = []
        for train in trains:
            reference_correct = tables[(train, test, REFERENCE)]["correct"].to_numpy()
            for competitor in competitors:
                competitor_correct = tables[(train, test, competitor)]["correct"].to_numpy()
                block.append({
                    "test_source": test,
                    "n_subjects": n,
                    "train_source": train,
                    "comparison": f"{REFERENCE} vs {competitor}",
                    "competitor": competitor,
                    "acc_reference": float(reference_correct.mean()),
                    "acc_competitor": float(competitor_correct.mean()),
                    "acc_diff_observed": float(reference_correct.mean() - competitor_correct.mean()),
                    **mcnemar_exact(reference_correct, competitor_correct),
                })
        adjusted, levels = holm_levels([row["p_mcnemar_exact"] for row in block])
        for row, p_adjusted, level in zip(block, adjusted, levels, strict=True):
            unadjusted = qh_interval(row["discordant_b"], row["discordant_c"], n, ALPHA)
            row["unadj_qh_point"] = unadjusted["qh_point_shrunk"]
            row["unadj_qh_lo"] = unadjusted["qh_lo"]
            row["unadj_qh_hi"] = unadjusted["qh_hi"]
            row.update(qh_interval(row["discordant_b"], row["discordant_c"], n, level))
            row["p_mcnemar_holm"] = p_adjusted
            row["holm_alpha"] = level
            row["rejects_after_holm"] = bool(p_adjusted < ALPHA)
            row["holm_interval_excludes_zero"] = bool(row["qh_lo"] > 0 or row["qh_hi"] < 0)
        rows.extend(block)
    return pd.DataFrame(rows)
