from typing import Type

from .base import (
    BaseMetric,
    ExactMatchMetric,
    F1Metric,
    PrecisionMetric,
    RecallMetric,
)
from src.registry import METRIC_REGISTRY, register_metric

# Import metric modules to trigger @register_metric decorators.
from . import (  # noqa: F401
    base,
    code_match,
    instruction_following,
    llm_judge,
    numeric_match,
)


def get_metric_class(name: str) -> Type[BaseMetric]:
    """
    Get the metric class by its name.

    Args:
        name: The name of the metric (e.g., "precision", "llm_judge").

    Returns:
        The corresponding BaseMetric subclass.

    Raises:
        ValueError: If the metric name is not found in the registry.
    """
    return METRIC_REGISTRY.get(name)


__all__ = [
    "BaseMetric",
    "PrecisionMetric",
    "RecallMetric",
    "ExactMatchMetric",
    "F1Metric",
    "METRIC_REGISTRY",
    "register_metric",
    "get_metric_class",
]
