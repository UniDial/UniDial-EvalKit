from typing import Dict, Type

from .base import BaseModel
from .openai import OpenAIModel
from .hipporag import HippoRAGModel
from .memoryos import MemoryOSModel
from .amem import AMemModel

# Registry mapping model type names to model classes
# Note: "openai" handles both standard OpenAI and Azure OpenAI via configuration
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "openai": OpenAIModel,
    "hipporag": HippoRAGModel,
    "memoryos": MemoryOSModel,
    "amem": AMemModel,
}

def get_model_class(model_type: str) -> Type[BaseModel]:
    """
    Get the model class by its type name.
    
    Args:
        model_type: The type of the model (e.g., "openai", "azure", "huggingface").
        
    Returns:
        The corresponding BaseModel subclass.
        
    Raises:
        ValueError: If the model type is not found in the registry.
    """
    model_type = model_type.lower()
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Model type '{model_type}' not found. Available types: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_type]

__all__ = [
    "BaseModel",
    "OpenAIModel",
    "HippoRAGModel",
    "MemoryOSModel",
    "AMemModel",
    "MODEL_REGISTRY",
    "get_model_class",
]

