"""Folador et al. 2021: HOG + {RFC, KNN, SVM_linear, CNN1D}.

The paper reports the SVM_linear head (the default). HOG params and classifier
hyperparameters are Folador's published numbers, copied verbatim. CPU-only.
"""

from typing import ClassVar

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from ..protocol import Pipeline

# Verbatim Folador HOG params.
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(16, 16),
    cells_per_block=(2, 2),
    transform_sqrt=True,
    block_norm="L1",
)


def quantify_image(image_2d: np.ndarray) -> np.ndarray:
    """Folador HOG feature extraction; copied verbatim."""
    from skimage.feature import hog

    return hog(image_2d, **HOG_PARAMS)


_VALID_CLASSIFIERS = {"RFC", "KNN", "SVM_linear", "CNN1D"}


class FoladorHOG(Pipeline):
    name: ClassVar[str] = "folador_hog"
    FREE_HYPERPARAMETERS: ClassVar[list[str]] = [
        "classifier",
        "hog_orientations",
        "hog_pixels_per_cell",
        "hog_cells_per_block",
    ]
    METHODS_SECTION_NOTES: ClassVar[str] = (
        "Folador et al. (2021) pipeline: HOG features (orientations=9, pixels_per_cell="
        "(16,16), cells_per_block=(2,2), transform_sqrt=True, block_norm='L1') applied to "
        "grayscale 256x256 spirals (the resize size that yields Folador's published "
        "8100-element feature vector), fed into one of four classifiers with Folador's "
        "published hyperparameters: RFC(n_estimators=500), KNN(n_neighbors=7, p=1, "
        "weights='distance'), SVM(kernel='linear'), or a 1D-CNN. The paper reports the "
        "SVM_linear head. The 1D-CNN is kept in Keras for faithful reproduction."
    )

    HOG_INPUT_SIZE: ClassVar[int] = 256  # Folador 2021 Sec 2.3 — fixed at 256x256, yields 8100 features.

    def __init__(self, classifier: str = "SVM_linear", batch_size: int = 32) -> None:
        if classifier not in _VALID_CLASSIFIERS:
            raise ValueError(f"classifier must be in {_VALID_CLASSIFIERS}; got {classifier!r}")
        self.classifier = classifier
        self.batch_size = batch_size
        self._model = None

    def _extract(self, dataset) -> tuple[np.ndarray, np.ndarray]:
        from torch.utils.data import DataLoader

        from . import loader_kwargs

        # HOG runs CPU-only on numpy; workers parallelize PIL decode + transform (Linux only).
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            **loader_kwargs(),
        )
        feats: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        for tensors, label_ints, _ in loader:
            # tensors: (B, C, H, W). HOG needs 2D grayscale — average channels if 3-ch.
            arr = tensors.numpy()
            if arr.shape[1] == 3:
                arr = arr.mean(axis=1)
            else:
                arr = arr[:, 0]
            for img2d in arr:
                # Folador 2021 operates on grayscale in [0, 1]; rescale per-image to avoid
                # negative values from upstream normalization (transform_sqrt=True would
                # otherwise produce NaN HOG features).
                lo, hi = float(img2d.min()), float(img2d.max())
                img_unit = (img2d - lo) / (hi - lo) if hi > lo else np.zeros_like(img2d)
                # Folador 2021 Sec 2.3 resizes to 256x256; HOG output dimensionality (8100)
                # depends on it. Resize internally so the pipeline stays faithful regardless
                # of which preprocessing recipe is paired upstream.
                if img_unit.shape != (self.HOG_INPUT_SIZE, self.HOG_INPUT_SIZE):
                    from PIL import Image

                    pil = Image.fromarray((img_unit * 255).astype(np.uint8), mode="L")
                    pil = pil.resize((self.HOG_INPUT_SIZE, self.HOG_INPUT_SIZE), Image.BILINEAR)
                    img_unit = np.asarray(pil, dtype=np.float32) / 255.0
                feats.append(quantify_image(img_unit))
            labels.append(np.asarray(label_ints))
        return np.stack(feats, axis=0), np.concatenate(labels, axis=0)

    def fit(self, dataset, *, seed: int) -> None:
        X, y = self._extract(dataset)
        if self.classifier == "RFC":
            self._model = RandomForestClassifier(n_estimators=500, random_state=seed)
        elif self.classifier == "KNN":
            self._model = KNeighborsClassifier(n_neighbors=7, p=1, weights="distance")
        elif self.classifier == "SVM_linear":
            self._model = SVC(kernel="linear", probability=True, random_state=seed)
        else:
            self._model = _build_cnn1d_keras(input_dim=X.shape[1], seed=seed)
            self._model.fit(X[..., None], y, epochs=10, batch_size=24, verbose=0)
            return
        self._model.fit(X, y)

    def predict_proba(self, dataset) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("predict_proba called before fit")
        X, _ = self._extract(dataset)
        if self.classifier == "CNN1D":
            p = self._model.predict(X[..., None], verbose=0).reshape(-1)
            return np.stack([1 - p, p], axis=1)
        return self._model.predict_proba(X)


def _build_cnn1d_keras(input_dim: int, seed: int):
    """Folador 1D-CNN architecture, copied verbatim."""
    import os

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers

    tf.random.set_seed(seed)
    reg = regularizers.l2(1e-4)
    model = models.Sequential([
        layers.Input(shape=(input_dim, 1)),
        layers.Conv1D(8, kernel_size=23, activation="relu", kernel_regularizer=reg),
        layers.Dropout(0.4),
        layers.Conv1D(16, kernel_size=23, activation="relu", kernel_regularizer=reg),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(32, activation="relu", kernel_regularizer=reg),
        layers.Dropout(0.3),
        layers.Dense(16, activation="relu", kernel_regularizer=reg),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid", kernel_regularizer=reg),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="binary_crossentropy", metrics=["accuracy"]
    )
    return model
