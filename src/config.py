"""
Evaluation Pipeline Configuration (EvalPipelineConfig)

A CLI-decoupled dataclass that can be constructed directly in scripts and notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class EvalPipelineConfig:
    """Global configuration for the evaluation pipeline."""

    # Dataset
    dataset: str = "mt_eval"
    raw_data_dir: str = "./raw_data/MT-Eval"
    processed_data_dir: str = "./data"
    output_dir: str = "./output"
    require_alternative_roles: bool = False

    # Generation model
    model_type: str = "openai"
    model_name: str = "deepseek-ai/DeepSeek-V3.2"
    temperature: float = 0.7
    max_tokens: int = 1024

    # Judge model
    judge_model_type: str = "openai"
    judge_model_name: str = "gpt-4.1-2025-04-14"

    # Embedding model for Agents
    embedding_model_name: str = "text-embedding-ada-002"
    
    # Common
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    parallel: int = 4

    # Task control
    do_generation: bool = False
    do_evaluation: bool = False

    # Aggregation
    agg_by_metric: bool = False  # generally true for instruction_following datasets
    agg_turn_stat: str = "mean"
    agg_dialog_stat: str = "min"
    agg_dataset_level: str = "dialog"

    # ---------- derived paths (auto-computed) ----------
    @property
    def gen_output_dir(self) -> Path:
        return Path(self.output_dir) / self.dataset / self.model_name / "generated"

    @property
    def eval_output_dir(self) -> Path:
        return Path(self.output_dir) / self.dataset / self.model_name / "eval_details"

    @property
    def summary_output_path(self) -> Path:
        return Path(self.output_dir) / self.dataset / self.model_name / "summary.json"

