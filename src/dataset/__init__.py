from typing import Type

from .base import BenchmarkDataset
from src.registry import DATASET_REGISTRY, register_dataset

# Import dataset modules to trigger @register_dataset decorators.
from . import (  # noqa: F401
    amemgym,
    locomo,
    longmemeval,
    mathchat,
    memorycode,
    mt_bench_101,
    mt_eval,
    multi_challenge,
    multi_if,
    personamem,
    safedialbench,
)


def get_dataset_class(name: str) -> Type[BenchmarkDataset]:
    """
    Get the dataset class by its name (benchmark_id).

    Args:
        name: The benchmark_id of the dataset.

    Returns:
        The corresponding BenchmarkDataset class.

    Raises:
        ValueError: If the dataset name is not found in the registry.
    """
    return DATASET_REGISTRY.get(name)


__all__ = [
    "BenchmarkDataset",
    "DATASET_REGISTRY",
    "register_dataset",
    "get_dataset_class",
]
