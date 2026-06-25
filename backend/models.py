"""
Request / Response schemas for the EvalKit API.
"""
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class TaskStatus(str, Enum):
    idle = "idle"
    running = "running"
    completed = "completed"
    failed = "failed"


class TaskCreateRequest(BaseModel):
    # ── Dataset ──────────────────────────────────────────────────────────
    dataset: str
    raw_data_dir: str
    processed_data_dir: str = "./data"
    output_dir: str = "./output"

    # ── Generation model ─────────────────────────────────────────────────
    model_type: str
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024

    # ── Judge model (used by LLM-as-judge metrics) ───────────────────────
    judge_model_type: str = "openai"
    judge_model_name: str = "gpt-4.1-2025-04-14"

    # ── Execution ────────────────────────────────────────────────────────
    parallel: int = 4
    do_generation: bool = True
    do_evaluation: bool = True

    # ── Aggregation ──────────────────────────────────────────────────────
    agg_by_metric: bool = False
    agg_turn_stat: str = "mean"
    agg_dialog_stat: str = "min"
    agg_dataset_level: str = "dialog"


class ProgressSnapshot(BaseModel):
    """Sent over WebSocket every second while a task is running."""
    status: str          # TaskStatus value
    phase: str           # "idle" | "data" | "generation" | "evaluation" | "aggregation" | "done"
    total: int
    completed: int
    logs: List[str]
    error: Optional[str] = None
