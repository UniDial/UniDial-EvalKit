from typing import Dict, Type

from .base import BaseModel
from .openai import OpenAIModel

# Lazy-loaded model classes
_LAZY_MODEL_MAP = {
    "hipporag": (".hipporag_agent", "HippoRAGModel"),
    "memoryos": (".memoryos_agent", "MemoryOSModel"),
    "amem": (".amem_agent", "AMemModel"),
    "lightmem": (".lightmem_agent", "LightMemModel"),
    "rfmem": (".rfmem_agent", "RFMemModel"),
    "mempalace": (".mempalace_agent", "MempalaceModel"),
}

# Registry mapping model type names to model classes
# Note: "openai" handles both standard OpenAI and Azure OpenAI via configuration
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "openai": OpenAIModel,
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
    if model_type in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_type]
    if model_type in _LAZY_MODEL_MAP:
        module_name, class_name = _LAZY_MODEL_MAP[model_type]
        import importlib
        module = importlib.import_module(module_name, package=__name__)
        cls = getattr(module, class_name)
        MODEL_REGISTRY[model_type] = cls
        return cls
    raise ValueError(f"Model type '{model_type}' not found. Available types: {list(MODEL_REGISTRY.keys()) + list(_LAZY_MODEL_MAP.keys())}")

def __getattr__(name: str):
    for _, (module_name, class_name) in _LAZY_MODEL_MAP.items():
        if name == class_name:
            import importlib
            module = importlib.import_module(module_name, package=__name__)
            return getattr(module, class_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BaseModel",
    "OpenAIModel",
    "HippoRAGModel",
    "MemoryOSModel",
    "AMemModel",
    "MempalaceModel",
    "MODEL_REGISTRY",
    "get_model_class",
]
