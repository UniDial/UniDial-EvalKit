from typing import Dict, Type

from .base import BenchmarkDataset
from .mt_eval import MTEvalDataset
from .multi_challenge import MultiChallengeDataset
from .mt_bench_101 import MTBench101Dataset
from .multi_if import MultiIFDataset
from .locomo import LoCoMoDataset
from .longmemeval import LongMemEvalDataset
from .personamem import PersonaMemDataset
from .mathchat import MathChatDataset
from .memorycode import MemoryCodeDataset
from .safedialbench import SafeDialBenchDataset

# Registry mapping dataset names (benchmark_id) to dataset classes
DATASET_REGISTRY: Dict[str, Type[BenchmarkDataset]] = {
    MTEvalDataset.benchmark_id: MTEvalDataset,
    MultiChallengeDataset.benchmark_id: MultiChallengeDataset,
    MTBench101Dataset.benchmark_id: MTBench101Dataset,
    MultiIFDataset.benchmark_id: MultiIFDataset,
    LoCoMoDataset.benchmark_id: LoCoMoDataset,
    LongMemEvalDataset.benchmark_id: LongMemEvalDataset,
    PersonaMemDataset.benchmark_id: PersonaMemDataset,
    MathChatDataset.benchmark_id: MathChatDataset,
    MemoryCodeDataset.benchmark_id: MemoryCodeDataset,
    SafeDialBenchDataset.benchmark_id: SafeDialBenchDataset,
}

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
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{name}' not found. Available datasets: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name]

__all__ = [
    "BenchmarkDataset",
    "MTEvalDataset",
    "DATASET_REGISTRY",
    "get_dataset_class",
]

