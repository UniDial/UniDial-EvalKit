"""
Central registry utilities for datasets, metrics, and models.

Usage:
    @register_dataset()
    class MyDataset(BenchmarkDataset):
        benchmark_id = "my_dataset"

    @register_metric("my_metric")
    class MyMetric(BaseMetric):
        ...

    @register_model("my_model")
    class MyModel(BaseModel):
        ...
"""

from __future__ import annotations

from typing import Callable, Dict, Iterator, MutableMapping, Optional, Type, TypeVar

T = TypeVar("T")


class Registry(MutableMapping[str, Type]):
    """A name-to-class registry with decorator-based registration."""

    def __init__(
        self,
        kind: str,
        *,
        id_attr: Optional[str] = None,
        normalize_key: bool = False,
    ) -> None:
        self._kind = kind
        self._id_attr = id_attr
        self._normalize_key = normalize_key
        self._registry: Dict[str, Type] = {}

    def register(self, name: Optional[str] = None) -> Callable[[Type[T]], Type[T]]:
        def decorator(cls: Type[T]) -> Type[T]:
            key = name
            if key is None and self._id_attr is not None:
                key = getattr(cls, self._id_attr, None)
            if not key:
                raise ValueError(
                    f"Cannot register {self._kind} '{cls.__name__}': "
                    f"provide a name or set class attribute '{self._id_attr}'."
                )
            if self._normalize_key:
                key = key.lower()
            if key in self._registry and self._registry[key] is not cls:
                raise ValueError(
                    f"Duplicate {self._kind} registration for '{key}': "
                    f"{self._registry[key].__name__} vs {cls.__name__}"
                )
            self._registry[key] = cls
            return cls

        return decorator

    def get(self, name: str) -> Type:
        key = name.lower() if self._normalize_key else name
        if key not in self._registry:
            raise ValueError(
                f"{self._kind.capitalize()} '{name}' not found. "
                f"Available: {list(self._registry.keys())}"
            )
        return self._registry[key]

    def __getitem__(self, key: str) -> Type:
        return self.get(key)

    def __setitem__(self, key: str, value: Type) -> None:
        self._registry[key] = value

    def __delitem__(self, key: str) -> None:
        del self._registry[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._registry)

    def __len__(self) -> int:
        return len(self._registry)

    def keys(self):
        return self._registry.keys()

    def values(self):
        return self._registry.values()

    def items(self):
        return self._registry.items()


DATASET_REGISTRY = Registry("dataset", id_attr="benchmark_id")
register_dataset = DATASET_REGISTRY.register

METRIC_REGISTRY = Registry("metric", id_attr="metric_name")
register_metric = METRIC_REGISTRY.register

MODEL_REGISTRY = Registry("model", id_attr="model_type", normalize_key=True)
register_model = MODEL_REGISTRY.register
