"""Dataset loaders package for Parkinson Model Training."""

from collections.abc import Callable
from pathlib import Path

from parkinson_model_train.data import ImageRecord
from parkinson_model_train.data.datasets.hemato import load_hemato_records
from parkinson_model_train.data.datasets.niats_folador import load_niats_folador_records
from parkinson_model_train.data.datasets.utils import load_image_tree_records

DatasetLoader = Callable[[Path], list[ImageRecord]]

DATASET_LOADERS: dict[str, DatasetLoader] = {
    "hemato": load_hemato_records,
    "image-tree": load_image_tree_records,
    "niats-folador": load_niats_folador_records,
}

__all__ = [
    "DATASET_LOADERS",
    "DatasetLoader",
    "load_hemato_records",
    "load_image_tree_records",
    "load_niats_folador_records",
]
