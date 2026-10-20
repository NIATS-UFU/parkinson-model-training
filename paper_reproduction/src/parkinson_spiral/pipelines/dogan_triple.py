"""Dogan 2024 DenseNet201 + Xception + NASNetMobile concat -> NCA -> linear SVM.

Faithful reproduction across two ML frameworks: DenseNet201 (torchvision) and
Xception (timm) supply PyTorch GAP features; NASNetMobile is loaded from
tf.keras.applications (the only place it ships) and its 1056-d GAP features are
concatenated in numpy.

Two fine-tune modes:
- fine_tune=False (paper catalog): all three public ImageNet backbones are frozen.
  This bounded M4 reproduction is explicit because TensorFlow 2.21 exposes no
  Apple Metal device for literal NASNetMobile fine-tuning.
- fine_tune=True (optional): Adam(default lr=1e-3), 20 epochs per PyTorch
  backbone, batch=32; NASNetMobile still remains frozen.
"""

from typing import ClassVar

import numpy as np
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from ..protocol import Pipeline
from .folador_hog import _stable_sigmoid

FINE_TUNE_EPOCHS = 20
FINE_TUNE_LR = 1e-3  # Dogan didn't state - Adam default
FINE_TUNE_BATCH = 32
NCA_RETENTION = 0.17  # Dogan section 5
INPUT_SIZE = 224  # Dogan's spec; both DenseNet/Xception are ImageNet-default 224x224
# NCA's optimiser stops at a different iteration when its BLAS products are split across a
# different number of threads, which moves the fitted projection and every downstream
# probability. Fixing the count removes most of the dependence on how many cores the machine
# has, but not all of it: in testing with 4 and 8 cores visible on one CPU model, a few results
# still differed. Reproducing a result exactly needs the same core count, CPU model and BLAS build.
NCA_THREADS = 8


class DoganTriple(Pipeline):
    name: ClassVar[str] = "dogan_triple"
    MODEL_WEIGHT_IDENTIFIERS: ClassVar[dict[str, str]] = {
        "densenet201": "torchvision:DenseNet201_Weights.IMAGENET1K_V1",
        "xception": "timm:xception:pretrained",
        "nasnet_mobile": "tf.keras.applications.NASNetMobile:imagenet",
    }
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
        "average pooling. The paper catalog uses all three as frozen public feature "
        "extractors because TensorFlow 2.21 exposes no Apple Metal device for literal "
        "NASNetMobile fine-tuning. Optional fine-tuning is available for DenseNet201 and "
        "Xception only, so either mode is a bounded reproduction rather than literal "
        "three-backbone fine-tuning. The three "
        "GAP feature vectors (1920 + 2048 + 1056 = 5024-d) are concatenated, passed through "
        "NeighborhoodComponentsAnalysis retaining 17% of the dimensionality (Dogan section 5), "
        "and classified with a linear SVM (Dogan did not specify the kernel; we use linear "
        "because of the high-dimensional concatenated features). Augmentation is applied "
        "after the released or grouped split; the source paper's augmentation-before-split "
        "result is treated as an unverified upper bound because its split is not published."
    )
    _BACKBONE_CACHE: ClassVar[dict[str, tuple[list, object, object]]] = {}
    _BACKBONE_CHECKSUMS: ClassVar[dict[str, dict[str, str]]] = {}
    _FEATURE_CACHE: ClassVar[dict[tuple, tuple[np.ndarray, np.ndarray]]] = {}
    _FIT_CACHE: ClassVar[dict[tuple, tuple[NeighborhoodComponentsAnalysis, SVC]]] = {}

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
        self._seed = 0

    def _build_backbones(self):
        import timm
        import torch
        from torchvision.models import DenseNet201_Weights, densenet201

        from . import accelerator_device

        self._device = accelerator_device()
        self._use_amp = self._device == "cuda"
        if not self.fine_tune and self._device in self._BACKBONE_CACHE:
            cached_torch, cached_nasnet, cached_preprocess = self._BACKBONE_CACHE[self._device]
            self._torch_backbones = cached_torch
            self._keras_nasnet = cached_nasnet
            self._keras_preprocess = cached_preprocess
            self._model_weight_checksums = self._BACKBONE_CHECKSUMS[self._device]
            return

        dn = densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)
        dn.classifier = torch.nn.Identity()  # gives GAP'd features
        xc = timm.create_model("xception", pretrained=True, num_classes=0, global_pool="avg")
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
        from ..weight_checksums import keras_model_sha256, torch_module_sha256

        self._model_weight_checksums = {
            "densenet201": torch_module_sha256(dn),
            "xception": torch_module_sha256(xc),
            "nasnet_mobile": keras_model_sha256(self._keras_nasnet),
        }
        self._BACKBONE_CHECKSUMS[self._device] = self._model_weight_checksums
        if not self.fine_tune:
            self._BACKBONE_CACHE[self._device] = (
                self._torch_backbones,
                self._keras_nasnet,
                self._keras_preprocess,
            )

    def _fine_tune_torch_backbones(self, dataset, seed: int):
        import torch
        from torch.utils.data import DataLoader
        from tqdm.auto import tqdm

        loss_fn = torch.nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler("cuda", enabled=self._use_amp)

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
                # forbid_fork: this pipeline's NCA does a large BLAS solve; a forked
                # DataLoader worker would poison OpenBLAS and deadlock it on Linux.
                **loader_kwargs(forbid_fork=True),
            )
            wrapped.train()
            for _epoch in tqdm(range(self.epochs), desc="dogan-ft-epoch", leave=False):
                for tensors, labels, _ in loader:
                    tensors = tensors.to(self._device, non_blocking=True)
                    labels = labels.to(self._device, non_blocking=True).long()
                    optim.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=self._use_amp):
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

    def _dataset_cache_key(self, dataset) -> tuple:
        image_ids = tuple(dataset.manifest["image_id"].astype(str))
        recipe_name = dataset.recipe.meta.name
        return (
            self._device,
            recipe_name,
            bool(dataset.is_train),
            bool(dataset.augment_then_split),
            self._seed,
            image_ids,
        )

    def _extract(self, dataset) -> tuple[np.ndarray, np.ndarray]:
        import torch
        from torch.utils.data import DataLoader

        from . import loader_kwargs

        cache_key = self._dataset_cache_key(dataset)
        if not self.fine_tune and cache_key in self._FEATURE_CACHE:
            return self._FEATURE_CACHE[cache_key]

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=(self._device == "cuda"),
            # forbid_fork: keep every Dogan DataLoader single-process on Linux so no
            # fork poisons OpenBLAS before the NCA BLAS solve (see loader_kwargs).
            **loader_kwargs(forbid_fork=True),
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
        result = np.concatenate(all_feats, axis=0), np.concatenate(all_labels, axis=0)
        if not self.fine_tune:
            self._FEATURE_CACHE[cache_key] = result
        return result

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
        # Use direct model call instead of predict() to bypass TF's tf.data.Dataset
        # prefetch pipeline, which deadlocks on macOS when TF and PyTorch MPS both
        # hold system synchronization resources.
        import tensorflow as tf

        return self._keras_nasnet(tf.constant(arr), training=False).numpy()

    def fit(self, dataset, *, seed: int) -> None:
        self._seed = seed
        self._build_backbones()
        import torch

        # A first build draws from torch's generator while constructing the layers; a cached
        # build draws nothing. Reseeding here gives the training augmentations the same random
        # stream either way, so a fit no longer depends on whether it ran first in the process.
        torch.manual_seed(seed)
        if self._keras_nasnet is not None:
            import tensorflow as tf

            tf.keras.utils.set_random_seed(seed)
            tf.config.experimental.enable_op_determinism()
        if self.fine_tune:
            self._fine_tune_torch_backbones(dataset, seed)
        fit_key = (
            self._dataset_cache_key(dataset),
            self.ncomponents_pct,
            self.svm_kernel,
            self.svm_C,
        )
        if not self.fine_tune and fit_key in self._FIT_CACHE:
            self._nca, self._svm = self._FIT_CACHE[fit_key]
            self.fit_was_cached = True
            return
        self.fit_was_cached = False
        X, y = self._extract(dataset)
        n_components = max(1, int(round(self.ncomponents_pct * X.shape[1])))
        self._nca = NeighborhoodComponentsAnalysis(n_components=n_components, random_state=seed)
        with threadpool_limits(limits=NCA_THREADS):
            X_nca = self._nca.fit_transform(X, y)
        self._svm = SVC(kernel=self.svm_kernel, C=self.svm_C, random_state=seed)
        self._svm.fit(X_nca, y)
        if not self.fine_tune:
            self._FIT_CACHE[fit_key] = (self._nca, self._svm)

    def predict_proba(self, dataset) -> np.ndarray:
        if self._svm is None or self._nca is None:
            raise RuntimeError("predict_proba called before fit")
        X, _ = self._extract(dataset)
        with threadpool_limits(limits=NCA_THREADS):
            X_nca = self._nca.transform(X)
        margin = self._svm.decision_function(X_nca)
        p = _stable_sigmoid(margin)
        return np.stack([1 - p, p], axis=1)
