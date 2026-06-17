"""Huang 2024 VGG19 + rot+flip + cosine annealing.

Hyperparameters quoted verbatim from Huang et al. 2024 §9. Parameters Huang did
not specify are defaulted with explicit justification in METHODS_SECTION_NOTES.
This pipeline fine-tunes a full VGG19 end-to-end and requires a GPU in practice.
"""

from typing import ClassVar

import numpy as np

from ..protocol import Pipeline

# Verbatim from Huang §9
LR = 1e-4
EPOCHS = 50
BATCH_SIZE = 32
T_0 = 10  # CosineAnnealingWarmRestarts; Huang did not specify — default
T_MULT = 2  # default
ETA_MIN = 1e-6  # default


class HuangVGG19(Pipeline):
    name: ClassVar[str] = "huang_vgg19"
    FREE_HYPERPARAMETERS: ClassVar[list[str]] = [
        "lr",
        "batch_size",
        "epochs",
        "scheduler",
        "T_0",
        "T_mult",
        "eta_min",
        "rotation_degrees",
        "image_size",
        "norm_mean",
        "norm_std",
        "freezing_layers",
    ]
    METHODS_SECTION_NOTES: ClassVar[str] = (
        "Huang et al. (2024) VGG19 recipe: ImageNet-pretrained VGG19 with the classifier "
        "head replaced by a 2-class linear layer. Trained with Adam(lr=1e-4), batch=32, "
        "50 epochs, CrossEntropyLoss, and CosineAnnealingWarmRestarts. Huang did not "
        "specify T_0, T_mult, eta_min — we default to PyTorch defaults (T_0=10, T_mult=2, "
        "eta_min=1e-6) and document. Normalization uses Huang's published mean=0.5/std=0.5 "
        "per channel (non-standard for ImageNet-pretrained VGG19, but kept faithful for "
        "the reproduction)."
    )

    def __init__(
        self,
        epochs: int = EPOCHS,
        lr: float = LR,
        batch_size: int = BATCH_SIZE,
        freeze_features: bool = False,
    ) -> None:
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.freeze_features = freeze_features
        self._model = None
        self._device = None

    def _build(self):
        import torch
        from torchvision.models import VGG19_Weights, vgg19

        model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, 2)
        if self.freeze_features:
            for p in model.features.parameters():
                p.requires_grad = False
        return model

    def fit(self, dataset, *, seed: int) -> None:
        import torch
        from torch.utils.data import DataLoader
        from tqdm.auto import tqdm

        torch.manual_seed(seed)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        use_amp = self._device == "cuda"
        self._model = self._build().to(self._device)

        optim = torch.optim.Adam((p for p in self._model.parameters() if p.requires_grad), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=T_0, T_mult=T_MULT, eta_min=ETA_MIN)
        loss_fn = torch.nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        from . import loader_kwargs

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=use_amp,
            **loader_kwargs(),
        )

        self._model.train()
        for _epoch in tqdm(range(self.epochs), desc="huang-epoch", leave=False):
            for tensors, labels, _ in loader:
                tensors = tensors.to(self._device, non_blocking=True)
                labels = labels.to(self._device, non_blocking=True).long()
                optim.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = self._model(tensors)
                    loss = loss_fn(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            sched.step()

    def predict_proba(self, dataset) -> np.ndarray:
        import torch
        from torch.utils.data import DataLoader

        if self._model is None:
            raise RuntimeError("predict_proba called before fit")
        from . import loader_kwargs

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=(self._device == "cuda"),
            **loader_kwargs(),
        )
        self._model.eval()
        out: list[np.ndarray] = []
        with torch.inference_mode():
            for tensors, _, _ in loader:
                tensors = tensors.to(self._device, non_blocking=True)
                probs = torch.softmax(self._model(tensors), dim=1)
                out.append(probs.detach().cpu().numpy())
        return np.concatenate(out, axis=0)
