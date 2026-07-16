"""Huang 2024 VGG19 + rot+flip + cosine annealing.

Hyperparameters quoted verbatim from Huang et al. 2024 §9. Parameters Huang did
not specify are defaulted with explicit justification in METHODS_SECTION_NOTES.
This pipeline fine-tunes a full VGG19 end-to-end and requires a GPU in practice.
"""

import math
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
GRAD_MAX_NORM = 1.0  # numerical safeguard for end-to-end VGG19 fine-tuning


def _uses_mixed_precision(device: str) -> bool:
    # CUDA has loss scaling below. MPS does not, so full precision avoids
    # underflowing the small gradients produced by end-to-end fine-tuning.
    return device == "cuda"


def _validate_gradient_norm_value(norm: float, *, recover_nonfinite: bool, epoch: int) -> None:
    if not math.isfinite(norm):
        if recover_nonfinite:
            return
        raise FloatingPointError(f"Huang VGG19 produced non-finite gradients in epoch {epoch + 1}")
    if norm == 0.0:
        raise RuntimeError(f"Huang VGG19 produced zero gradients in epoch {epoch + 1}; training would collapse")


def _clip_gradients(parameters, *, recover_nonfinite: bool, epoch: int) -> None:
    import torch

    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(
        trainable,
        max_norm=GRAD_MAX_NORM,
        error_if_nonfinite=False,
    )
    _validate_gradient_norm_value(float(norm), recover_nonfinite=recover_nonfinite, epoch=epoch)


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
        "gradient_max_norm",
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
        "the reproduction). Gradients are clipped to a maximum norm of 1.0 after AMP "
        "unscaling to prevent unstable end-to-end updates."
    )
    _LAST_FIT_KEY: ClassVar[tuple | None] = None
    _LAST_MODEL: ClassVar[object | None] = None
    _LAST_DEVICE: ClassVar[str | None] = None

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
        from . import accelerator_device

        self._device = accelerator_device()
        fit_key = (
            tuple(dataset.manifest["image_id"].astype(str)),
            dataset.recipe.meta.name,
            seed,
            self.epochs,
            self.lr,
            self.batch_size,
            self.freeze_features,
            self._device,
        )
        if self._LAST_FIT_KEY == fit_key and self._LAST_MODEL is not None:
            self._model = self._LAST_MODEL
            self._device = self._LAST_DEVICE
            return
        use_amp = _uses_mixed_precision(self._device)
        use_scaler = use_amp
        self._model = self._build().to(self._device)

        optim = torch.optim.Adam((p for p in self._model.parameters() if p.requires_grad), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=T_0, T_mult=T_MULT, eta_min=ETA_MIN)
        loss_fn = torch.nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        from . import loader_kwargs

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=use_scaler,
            **loader_kwargs(),
        )

        self._model.train()
        for epoch in tqdm(range(self.epochs), desc="huang-epoch", leave=False):
            epoch_had_update = False
            for tensors, labels, _ in loader:
                tensors = tensors.to(self._device, non_blocking=True)
                labels = labels.to(self._device, non_blocking=True).long()
                optim.zero_grad(set_to_none=True)
                with torch.amp.autocast(self._device, enabled=use_amp):
                    logits = self._model(tensors)
                    loss = loss_fn(logits, labels)
                if use_scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optim)
                    _clip_gradients(self._model.parameters(), recover_nonfinite=True, epoch=epoch)
                    scale_before_step = scaler.get_scale()
                    scaler.step(optim)
                    scaler.update()
                    epoch_had_update |= scaler.get_scale() >= scale_before_step
                else:
                    loss.backward()
                    _clip_gradients(self._model.parameters(), recover_nonfinite=False, epoch=epoch)
                    optim.step()
                    epoch_had_update = True
            if not epoch_had_update:
                raise FloatingPointError(f"Huang VGG19 produced no valid optimizer updates in epoch {epoch + 1}")
            sched.step()
        self.__class__._LAST_FIT_KEY = fit_key
        self.__class__._LAST_MODEL = self._model
        self.__class__._LAST_DEVICE = self._device

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
