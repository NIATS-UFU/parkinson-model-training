"""Recipe registry. Keyed by the paper's recipe names."""
from .base import Recipe, RecipeMeta
from .aug_none import AugNone
from .aug_huang import AugHuang
from .aug_dogan import AugDogan

REGISTRY: dict[str, type[Recipe]] = {
    "AUG-None": AugNone,
    "AUG-Huang": AugHuang,
    "AUG-Dogan": AugDogan,
}


def get_recipe(name: str, **kwargs) -> Recipe:
    if name not in REGISTRY:
        raise ValueError(f"unknown recipe {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


__all__ = ["Recipe", "RecipeMeta", "REGISTRY", "get_recipe", "AugNone", "AugHuang", "AugDogan"]
