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
    model_name: str = "deepseek-v3.2"
    # If None, do NOT pass temperature to the LLM backend (use backend default).
    temperature: Optional[float] = None
    max_tokens: Optional[int] = 1024

    # Judge model
    judge_model_type: str = "openai"
    judge_model_name: str = "gpt-4.1-2025-04-14"

    # Embedding model for Agents
    embedding_model_name: str = "all-MiniLM-L6-v2" # "text-embedding-ada-002" # a-mem: sentence-transformers/all-MiniLM-L6-v2
    
    # Common
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    parallel: int = 4

    # Task control
    do_generation: bool = False
    do_evaluation: bool = False
    
    # Output control flags
    save_agent_logs: bool = True
    save_llm_logs: bool = True

    # Aggregation
    agg_by_metric: bool = False  # generally true for instruction_following datasets
    agg_turn_stat: str = "mean"
    agg_dialog_stat: str = "min"
    agg_dataset_level: str = "dialog"

    # ---------- derived paths (auto-computed) ----------
    @property
    def model_name_last(self) -> str:
        """Use the last segment so '/' won't create nested dirs."""
        return (self.model_name or "").split("/")[-1]

    @property
    def gen_output_dir(self) -> Path:
        return (
            Path(self.output_dir)
            / self.dataset
            / Path(self.model_type + "-" + self.model_name_last)
            / "generated"
        )

    @property
    def eval_output_dir(self) -> Path:
        return (
            Path(self.output_dir)
            / self.dataset
            / Path(self.model_type + "-" + self.model_name_last)
            / "eval_details"
        )

    @property
    def summary_output_path(self) -> Path:
        return (
            Path(self.output_dir)
            / self.dataset
            / Path(self.model_type + "-" + self.model_name_last)
            / "summary.json"
        )

    @property
    def agent_logs_output_dir(self) -> Path:
        return (
            Path(self.output_dir)
            / self.dataset
            / Path(self.model_type + "-" + self.model_name_last)
            / "agent_logs"
        )
    
    

def apply_overrides(cfg: EvalPipelineConfig, **overrides) -> EvalPipelineConfig:
    """
    Apply overrides onto an existing config.

    Rule: only keys with value is not None will be applied (so CLI can omit args).
    """
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    return cfg

