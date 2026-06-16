"""Dogan 2024 DenseNet201 + Xception + NASNetMobile concat -> NCA -> linear SVM.

Faithful reproduction across two ML frameworks: DenseNet201 (torchvision) and
Xception (timm) supply PyTorch GAP features; NASNetMobile is loaded from
tf.keras.applications (the only place it ships) and its 1056-d GAP features are
concatenated in numpy.

Two fine-tune modes:
- fine_tune=True (faithful): Adam(default lr=1e-3), 20 epochs per backbone, batch=32.
  Fine-tunes only the PyTorch backbones; Keras NASNetMobile stays frozen (recorded
  in METHODS_SECTION_NOTES).
- fine_tune=False: all three backbones frozen, ImageNet weights only.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.svm import SVC

from ..protocol import Pipeline

FINE_TUNE_EPOCHS = 20
FINE_TUNE_LR = 1e-3  # Dogan didn't state - Adam default
FINE_TUNE_BATCH = 32
NCA_RETENTION = 0.17  # Dogan section 5
INPUT_SIZE = 224  # Dogan's spec; both DenseNet/Xception are ImageNet-default 224x224


class DoganTriple(Pipeline):
    name: ClassVar[str] = "dogan_triple"
    FREE_HYPERPARAMETERS: ClassVar[list[str]] = [
        "fine_tune",
        "lr",
        "batch_size",
        "epochs",
        "image_size",
        "ncomponents_pct",
        "svm_kernel",
        "svm_C",
        "rotation_set",
        "color_jitter_params",
        "augment_then_split",
    ]
    METHODS_SECTION_NOTES: ClassVar[str] = (
        "Dogan et al. (2024) hybrid pipeline reproduced faithfully across two ML frameworks. "
        "Three ImageNet-pretrained CNNs - DenseNet201 (torchvision), Xception (timm) and "
        "NASNetMobile (tf.keras.applications) - are used as feature extractors after global "
        "average pooling. We support two modes: faithful fine-tuning (Adam, 20 epochs, "
        "batch=32, default lr=1e-3) of the PyTorch backbones (Keras NASNetMobile stays "
        "frozen), and a frozen-features fallback for compute-constrained settings. The three "
        "GAP feature vectors (1920 + 2048 + 1056 = 5024-d) are concatenated, passed through "
        "NeighborhoodComponentsAnalysis retaining 17% of the dimensionality (Dogan section 5), "
        "and classified with a linear SVM (Dogan did not specify the kernel; we use linear "
        "because of the high-dimensional concatenated features). When augment_then_split=True, "
        "the dataset emits 8 deterministic rotation/flip copies per source image."
    )

    def __init__(
        self,
        fine_tune: bool = False,
        epochs: int = FINE_TUNE_EPOCHS,
        lr: float = FINE_TUNE_LR,
        batch_size: int = FINE_TUNE_BATCH,
        ncomponents_pct: float = NCA_RETENTION,
        svm_kernel: str = "linear",
        svm_C: float = 1.0,
    ) -> None:
        self.fine_tune = fine_tune
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.ncomponents_pct = ncomponents_pct
        self.svm_kernel = svm_kernel
        self.svm_C = svm_C
        self._torch_backbones: list = []  # [DenseNet201, Xception]
        self._keras_nasnet = None  # tf.keras NASNetMobile
        self._keras_preprocess = None
        self._nca: NeighborhoodComponentsAnalysis | None = None
        self._svm: SVC | None = None
        self._device = None
        self._use_amp = False

    def _build_backbones(self):
        import timm
        import torch
        from torchvision.models import DenseNet201_Weights, densenet201

        dn = densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)
        dn.classifier = torch.nn.Identity()  # gives GAP'd features

        xc = timm.create_model("xception", pretrained=True, num_classes=0, global_pool="avg")

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._use_amp = self._device == "cuda"
        self._torch_backbones = [m.eval().to(self._device) for m in (dn, xc)]
        if not self.fine_tune:
            for m in self._torch_backbones:
                for p in m.parameters():
                    p.requires_grad = False

        # NASNetMobile via Keras - this is the faithful Dogan choice.
        import os

        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        from tensorflow.keras.applications.nasnet import NASNetMobile, preprocess_input

        self._keras_nasnet = NASNetMobile(
            include_top=False, weights="imagenet", pooling="avg", input_shape=(INPUT_SIZE, INPUT_SIZE, 3)
        )
        self._keras_nasnet.trainable = False  # frozen; see METHODS_SECTION_NOTES
        self._keras_preprocess = preprocess_input

    def _fine_tune_torch_backbones(self, dataset, seed: int):
        import torch
        from torch.utils.data import DataLoader
        from tqdm.auto import tqdm

        loss_fn = torch.nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler(enabled=self._use_amp)

        for backbone in self._torch_backbones:
            head = torch.nn.Linear(self._feature_dim(backbone), 2).to(self._device)
            wrapped = torch.nn.Sequential(backbone, head)
            optim = torch.optim.Adam(wrapped.parameters(), lr=self.lr)
            from . import loader_kwargs

            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=(self._device == "cuda"),
                **loader_kwargs(),
            )
            wrapped.train()
            for _epoch in tqdm(range(self.epochs), desc="dogan-ft-epoch", leave=False):
                for tensors, labels, _ in loader:
                    tensors = tensors.to(self._device, non_blocking=True)
                    labels = labels.to(self._device, non_blocking=True).long()
                    optim.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=self._use_amp):
                        logits = wrapped(tensors)
                        loss = loss_fn(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optim)
                    scaler.update()
            backbone.eval()

    @staticmethod
    def _feature_dim(backbone) -> int:
        import torch

        with torch.inference_mode():
            x = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE, device=next(backbone.parameters()).device)
            return backbone(x).shape[-1]

    def _extract(self, dataset) -> tuple[np.ndarray, np.ndarray]:
        import torch
        from torch.utils.data import DataLoader

        from . import loader_kwargs

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=(self._device == "cuda"),
            **loader_kwargs(),
        )
        all_feats: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        with torch.inference_mode():
            for tensors, labels, _ in loader:
                # PyTorch backbones first.
                torch_tensors = tensors.to(self._device, non_blocking=True)
                torch_feats = [m(torch_tensors).detach().cpu().numpy() for m in self._torch_backbones]

                # Keras NASNetMobile: convert (B, C, H, W) torch tensor -> (B, H, W, C) numpy,
                # then apply Keras' channel-scaling preprocess_input and predict GAP features.
                nas_feats = self._keras_extract(tensors)

                concat = np.concatenate([*torch_feats, nas_feats], axis=1)
                all_feats.append(concat)
                all_labels.append(np.asarray(labels))
        return np.concatenate(all_feats, axis=0), np.concatenate(all_labels, axis=0)

    def _keras_extract(self, tensors) -> np.ndarray:
        """Run Keras NASNetMobile on a torch batch and return (B, 1056) GAP features."""
        # PreprocessedSpiralDataset emits ImageNet-normalized tensors. Un-normalize to
        # [0, 255] so Keras' preprocess_input (which expects [-1, 1] scaled from raw
        # pixels) sees the right input distribution.
        import numpy as np

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
        arr = tensors.detach().cpu().numpy().astype(np.float32)
        arr = arr * std + mean  # back to [0, 1]
        arr = np.clip(arr * 255.0, 0, 255)  # to [0, 255]
        arr = arr.transpose(0, 2, 3, 1)  # (B, H, W, C)
        arr = self._keras_preprocess(arr)
        return self._keras_nasnet.predict(arr, verbose=0)

    def fit(self, dataset, *, seed: int) -> None:
        self._build_backbones()
        if self.fine_tune:
            self._fine_tune_torch_backbones(dataset, seed)
        X, y = self._extract(dataset)
        n_components = max(1, int(round(self.ncomponents_pct * X.shape[1])))
        self._nca = NeighborhoodComponentsAnalysis(n_components=n_components, random_state=seed)
        X_nca = self._nca.fit_transform(X, y)
        self._svm = SVC(kernel=self.svm_kernel, C=self.svm_C, probability=True, random_state=seed)
        self._svm.fit(X_nca, y)

    def predict_proba(self, dataset) -> np.ndarray:
        if self._svm is None or self._nca is None:
            raise RuntimeError("predict_proba called before fit")
        X, _ = self._extract(dataset)
        return self._svm.predict_proba(self._nca.transform(X))
