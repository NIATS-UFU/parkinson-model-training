"""PreprocessedSpiralDataset: torch.utils.data.Dataset over the unified manifest.

Construction:
    PreprocessedSpiralDataset(manifest_df, recipe, is_train, augment_then_split=False)

In `augment_then_split=True` mode, len() == manifest_len × recipe.variants_per_image()
and __getitem__ decodes the linear index into (row, variant). This is only used
by the one Dogan fidelity cell that reproduces the literature's augment-before-split
order. Everywhere else, augmentation is applied after the split (the honest order).
"""
from __future__ import annotations

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..recipes.base import Recipe

CLASS_LABEL_TO_INT = {"healthy": 0, "parkinson": 1}


class PreprocessedSpiralDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        recipe: Recipe,
        is_train: bool,
        augment_then_split: bool = False,
        target_column: str = "class_label",
    ) -> None:
        if target_column != "class_label":
            raise ValueError("only the binary 'class_label' target is supported in this release")
        self.manifest = manifest.reset_index(drop=True)
        self.recipe = recipe
        self.is_train = is_train
        self.augment_then_split = augment_then_split
        self.target_column = target_column
        self._variants = recipe.variants_per_image() if augment_then_split else 1

    def __len__(self) -> int:
        return len(self.manifest) * self._variants

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        if self.augment_then_split:
            row_idx, variant_idx = divmod(idx, self._variants)
        else:
            row_idx, variant_idx = idx, 0
        row = self.manifest.iloc[row_idx]
        img = Image.open(row["file_path"])
        if self.augment_then_split:
            tensor = self.recipe.get_variant(img, variant_idx)
        else:
            img = self.recipe.pre_split_ops(img)
            tensor = (
                self.recipe.train_augment(img)
                if self.is_train
                else self.recipe.eval_transform(img)
            )
        label = CLASS_LABEL_TO_INT[row["class_label"]]
        return tensor, label, row["image_id"]
