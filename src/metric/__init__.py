from typing import Dict, Type

from .base import (
    BaseMetric,
    # HuggingFaceMetric,
    PrecisionMetric,
    RecallMetric,
    ExactMatchMetric,
    F1Metric
)
from .instruction_following import InstructionFollowingMetric
from .llm_judge import LLMJudge

# Registry mapping metric names to metric classes
METRIC_REGISTRY: Dict[str, Type[BaseMetric]] = {
    "precision": PrecisionMetric,
    "recall": RecallMetric,
    "exact_match": ExactMatchMetric,
    "f1_score": F1Metric,
    "instruction_following": InstructionFollowingMetric,
    "llm_judge": LLMJudge,
}

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
    if name not in METRIC_REGISTRY:
        raise ValueError(f"Metric '{name}' not found. Available metrics: {list(METRIC_REGISTRY.keys())}")
    return METRIC_REGISTRY[name]

__all__ = [
    "BaseMetric",
    # "HuggingFaceMetric",
    "PrecisionMetric",
    "RecallMetric",
    "ExactMatchMetric",
    "F1Metric",
    "InstructionFollowingMetric",
    "LLMJudge",
    "METRIC_REGISTRY",
    "get_metric_class",
]

