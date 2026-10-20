"""How far apart the datasets are in the frozen DINOv2 embedding space.

Reads the embeddings `scripts/paper.py export-embeddings` writes and never trains. Each participant
is the mean embedding of their spiral drawings, so a dataset is a cloud of participants.

- Distance. The squared sliced Wasserstein-2 distance (Nadjahi et al. 2020, "Statistical and
  topological properties of sliced probability divergences", NeurIPS): project both clouds onto
  random unit directions, take the 1-D squared Wasserstein-2 distance per direction (the mean
  squared gap between the two empirical quantile functions), and average over directions. Each
  dataset pair is measured on equal-size samples of participants, repeated over many draws, and
  the reported distance is the square root of the mean.
- Floor. The same measure between two disjoint halves of one dataset's participants: what two
  samples of the same population give.
- Null. The same measure after shuffling which dataset each participant belongs to.

The estimate does not settle as the sample grows at these sizes, so read the distances as an
ordering of the pairs, not as absolute magnitudes.
"""

import numpy as np
import pandas as pd

DATASETS = ("kaggle", "folador", "hemato")
PAIRS = (("kaggle", "folador"), ("kaggle", "hemato"), ("folador", "hemato"))
DISTANCE_SEED = 20260902
N_DIRECTIONS = 1000
N_DRAWS = 200
N_PER_SAMPLE = 14  # participants per sample, below the smallest dataset's 28
QUANTILE_GRID = np.linspace(0.5 / 512, 1 - 0.5 / 512, 512)


def feature_columns(embeddings: pd.DataFrame) -> list[str]:
    """The embedding columns f0, f1, ..., in numeric order."""
    columns = [c for c in embeddings.columns if c.startswith("f") and c[1:].isdigit()]
    if not columns:
        raise ValueError("no embedding columns (f0, f1, ...) found")
    return sorted(columns, key=lambda c: int(c[1:]))


def participant_clouds(embeddings: pd.DataFrame) -> dict[str, np.ndarray]:
    """Mean spiral embedding per participant, one array per dataset, participants in id order."""
    spirals = embeddings[embeddings.drawing_type == "spiral"]
    columns = feature_columns(spirals)
    missing = sorted(set(DATASETS) - set(spirals.dataset.unique()))
    if missing:
        raise ValueError(f"no spiral embeddings for {missing}")
    return {
        dataset: spirals[spirals.dataset == dataset].groupby("subject_id")[columns].mean().to_numpy(np.float64)
        for dataset in DATASETS
    }


def unit_directions(dimension: int, count: int, rng: np.random.Generator) -> np.ndarray:
    """`count` directions drawn uniformly on the unit sphere."""
    gaussian = rng.standard_normal((count, dimension))
    return gaussian / np.linalg.norm(gaussian, axis=1, keepdims=True)


def sliced_w2_squared(x: np.ndarray, y: np.ndarray, directions: np.ndarray) -> float:
    """Squared sliced Wasserstein-2 distance between two point clouds over fixed directions.

    Per direction the 1-D squared Wasserstein-2 distance is the integral of the squared gap between
    the two empirical quantile functions, evaluated on a fixed midpoint grid.
    """
    qx = np.quantile(directions @ x.T, QUANTILE_GRID, axis=1, method="inverted_cdf")
    qy = np.quantile(directions @ y.T, QUANTILE_GRID, axis=1, method="inverted_cdf")
    return float(((qx - qy) ** 2).mean(axis=0).mean())


def _summary(values: list[float]) -> dict:
    v = np.asarray(values, dtype=float)
    return {
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)),
        "p05": float(np.percentile(v, 5)),
        "p95": float(np.percentile(v, 95)),
        "n_draws": len(v),
    }


def _sample(rng: np.random.Generator, cloud: np.ndarray, n: int) -> np.ndarray:
    return cloud[rng.choice(len(cloud), n, replace=False)]


def dataset_distances(
    clouds: dict[str, np.ndarray],
    *,
    seed: int = DISTANCE_SEED,
    n_directions: int = N_DIRECTIONS,
    n_draws: int = N_DRAWS,
    n_per_sample: int = N_PER_SAMPLE,
) -> pd.DataFrame:
    """Between-dataset distances with their permutation null, and each dataset's floor.

    One row per dataset pair (kind "between") and per dataset (kind "within"). `distance` is the
    square root of the mean squared distance over the draws; the other columns describe the
    squared values. Every pair reuses the same directions and starts its draws from `seed`.
    """
    dimension = next(iter(clouds.values())).shape[1]
    directions = unit_directions(dimension, n_directions, np.random.default_rng(seed + 100))
    pooled = np.vstack([clouds[d] for d in DATASETS])
    cut = np.cumsum([0] + [len(clouds[d]) for d in DATASETS])

    rows = []
    for a, b in PAIRS:
        rng = np.random.default_rng(seed)
        observed = [
            sliced_w2_squared(_sample(rng, clouds[a], n_per_sample), _sample(rng, clouds[b], n_per_sample), directions)
            for _ in range(n_draws)
        ]
        rng = np.random.default_rng(seed + 99)
        ia, ib = DATASETS.index(a), DATASETS.index(b)
        null = []
        for _ in range(n_draws):
            order = rng.permutation(len(pooled))
            group_a, group_b = pooled[order[cut[ia] : cut[ia + 1]]], pooled[order[cut[ib] : cut[ib + 1]]]
            null.append(
                sliced_w2_squared(_sample(rng, group_a, n_per_sample), _sample(rng, group_b, n_per_sample), directions)
            )
        summary, null_summary = _summary(observed), _summary(null)
        rows.append({
            "kind": "between",
            "a": a,
            "b": b,
            "distance": float(np.sqrt(summary["mean"])),
            **{f"sq_{key}": value for key, value in summary.items()},
            "null_sq_mean": null_summary["mean"],
            "null_sq_p95": null_summary["p95"],
            "exceeds_null_p95": bool(summary["mean"] > null_summary["p95"]),
        })

    for dataset in DATASETS:
        rng = np.random.default_rng(seed + 1)
        cloud = clouds[dataset]
        values = []
        for _ in range(n_draws):
            # Two disjoint halves of the participants; each half must hold a full sample.
            order = rng.permutation(len(cloud))
            half = len(cloud) // 2
            first, second = np.sort(order[:half]), np.sort(order[half:])
            if len(first) < n_per_sample or len(second) < n_per_sample:
                raise ValueError(f"{dataset}: {len(cloud)} participants cannot give two halves of {n_per_sample}")
            values.append(
                sliced_w2_squared(
                    cloud[rng.choice(first, n_per_sample, replace=False)],
                    cloud[rng.choice(second, n_per_sample, replace=False)],
                    directions,
                )
            )
        summary = _summary(values)
        rows.append({
            "kind": "within",
            "a": dataset,
            "b": dataset,
            "distance": float(np.sqrt(summary["mean"])),
            **{f"sq_{key}": value for key, value in summary.items()},
        })

    frame = pd.DataFrame(rows)
    floor_p95 = frame[frame.kind == "within"].set_index("a")["sq_p95"]
    frame["exceeds_floor_p95"] = pd.array(
        [
            row.sq_mean > max(floor_p95[row.a], floor_p95[row.b]) if row.kind == "between" else None
            for row in frame.itertuples()
        ],
        dtype="boolean",
    )
    frame["exceeds_null_p95"] = frame["exceeds_null_p95"].astype("boolean")
    return frame


def tsne_coordinates(embeddings: pd.DataFrame, *, seed: int = DISTANCE_SEED) -> pd.DataFrame:
    """2-D t-SNE map of every spiral embedding (PCA to 50 dimensions first), for the figure.

    Gaps between clusters in a t-SNE map are not distances; `dataset_distances` measures those.
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    spirals = embeddings[embeddings.drawing_type == "spiral"].reset_index(drop=True)
    reduced = PCA(n_components=50, random_state=seed).fit_transform(
        spirals[feature_columns(spirals)].to_numpy(np.float64)
    )
    coordinates = TSNE(
        n_components=2, perplexity=30, init="pca", random_state=seed, learning_rate="auto"
    ).fit_transform(reduced)
    out = spirals[["image_id", "subject_id", "dataset", "class_label"]].copy()
    out["x"], out["y"] = coordinates[:, 0], coordinates[:, 1]
    return out
