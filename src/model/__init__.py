import importlib
from typing import Dict, Type

from .base import BaseModel
from src.registry import MODEL_REGISTRY, register_model

from . import openai  # noqa: F401

# Lazy-loaded model modules (heavy / optional dependencies).
_LAZY_MODEL_MODULES: Dict[str, str] = {
    "hipporag": ".hipporag_agent",
    "memoryos": ".memoryos_agent",
    "amem": ".amem_agent",
    "lightmem": ".lightmem_agent",
    "rfmem": ".rfmem_agent",
    "mempalace": ".mempalace_agent",
}


def _ensure_model_registered(model_type: str) -> None:
    if model_type in MODEL_REGISTRY:
        return
    module_path = _LAZY_MODEL_MODULES.get(model_type)
    if module_path is None:
        return
    importlib.import_module(module_path, package=__name__)


def get_model_class(model_type: str) -> Type[BaseModel]:
    """
    Get the model class by its type name.

    Args:
        model_type: The type of the model (e.g., "openai", "lightmem").

    Returns:
        The corresponding BaseModel subclass.

    Raises:
        ValueError: If the model type is not found in the registry.
    """
    model_type = model_type.lower()
    _ensure_model_registered(model_type)
    if model_type not in MODEL_REGISTRY:
        available = sorted(set(MODEL_REGISTRY.keys()) | set(_LAZY_MODEL_MODULES.keys()))
        raise ValueError(
            f"Model type '{model_type}' not found. Available types: {available}"
        )
    return MODEL_REGISTRY[model_type]


def __getattr__(name: str):
    for model_type, module_path in _LAZY_MODEL_MODULES.items():
        module = importlib.import_module(module_path, package=__name__)
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseModel",
    "OpenAIModel",
    "HippoRAGModel",
    "MemoryOSModel",
    "AMemModel",
    "MempalaceModel",
    "MODEL_REGISTRY",
    "register_model",
    "get_model_class",
]
