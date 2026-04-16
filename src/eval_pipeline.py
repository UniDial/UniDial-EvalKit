"""
Modular Evaluation Pipeline

Splits the evaluation workflow into four independent phases:
  1. DataPhase        — Data preprocessing & loading
  2. GenerationPhase  — Model inference / generation
  3. EvaluationPhase  — Metric evaluation
  4. AggregationPhase — Result aggregation

Each phase can be invoked independently, or chained via EvalPipeline.run().

This file is the unified entry point; external usage only needs:
    from src.eval_pipeline import EvalPipeline, EvalPipelineConfig
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.dataset import get_dataset_class, BenchmarkDataset
from src.dataset.schema import Dialog
from src.metric import get_metric_class, METRIC_REGISTRY
from src.model import get_model_class, BaseModel

# Re-export: Config & Phases
from src.config import EvalPipelineConfig
from src.phases import (
    DataPhase,
    GenerationPhase,
    EvaluationPhase,
    AggregationPhase,
)

logger = logging.getLogger(__name__)

__all__ = [
    # Config
    "EvalPipelineConfig",
    # Phases
    "DataPhase",
    "GenerationPhase",
    "EvaluationPhase",
    "AggregationPhase",
    # Factories
    "create_generation_model",
    "create_metrics_map",
    # Orchestrator
    "EvalPipeline",
]


# ---------------------------------------------------------------------------
# Model & Metric Factories (centralized initialization logic)
# ---------------------------------------------------------------------------

def create_generation_model(cfg: EvalPipelineConfig) -> BaseModel:
    """Create a generation model instance based on the config."""
    logger.info(f"Initializing generation model: {cfg.model_name} (type={cfg.model_type})")
    ModelClass = get_model_class(cfg.model_type)
    return ModelClass(
        model_name=cfg.model_name,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        dataset_name=cfg.dataset,
        embedding_model_name=cfg.embedding_model_name,
        save_llm_logs=cfg.save_llm_logs,
        save_agent_logs=cfg.save_agent_logs,
        agent_logs_output_dir=cfg.agent_logs_output_dir
    )


def create_metrics_map(
    dataset: BenchmarkDataset,
    cfg: EvalPipelineConfig,
) -> Dict[str, Any]:
    """Initialize all required Metric instances based on dataset config and eval parameters."""
    metric_configs = dataset.metric_configs()

    # Support both list and dict return formats
    if isinstance(metric_configs, list):
        metric_configs = {name: {} for name in metric_configs}

    metrics_map: Dict[str, Any] = {}
    for name, config in metric_configs.items():
        if name not in METRIC_REGISTRY:
            logger.warning(f"Metric '{name}' not found in registry. Skipping.")
            continue

        if name == "llm_judge":
            JudgeModelClass = get_model_class(cfg.judge_model_type)
            judge_model = JudgeModelClass(
                model_name=cfg.judge_model_name,
                api_key=cfg.api_key,
                base_url=cfg.base_url,
                save_llm_logs=cfg.save_llm_logs,
            )
            metrics_map[name] = get_metric_class(name)(
                llm_client=judge_model, dataset=dataset, **config
            )
        else:
            metrics_map[name] = get_metric_class(name)(**config)

    return metrics_map


# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------

class EvalPipeline:
    """
    Evaluation pipeline orchestrator.

    Usage 1 — Run all at once:
        pipeline = EvalPipeline(cfg)
        pipeline.run()

    Usage 2 — Run phase by phase:
        pipeline = EvalPipeline(cfg)
        dialogs = pipeline.prepare_data()
        generated = pipeline.run_generation(dialogs)
        results  = pipeline.run_evaluation(generated)
        summary  = pipeline.run_aggregation(results)
    """

    def __init__(self, cfg: EvalPipelineConfig) -> None:
        self.cfg = cfg
        self.dataset: BenchmarkDataset = self._init_dataset()

    # ---- Initialization ----
    def _init_dataset(self) -> BenchmarkDataset:
        logger.info(f"Initialize dataset: {self.cfg.dataset}")
        DatasetClass = get_dataset_class(self.cfg.dataset)
        return DatasetClass()

    # ---- Phase Methods ----
    def prepare_data(self) -> List[Dialog]:
        """Phase 1: Data preprocessing & loading."""
        return DataPhase.run(self.dataset, self.cfg)

    def run_generation(self, dialogs: List[Dialog]) -> List[Dialog]:
        """Phase 2: Model generation."""
        model = create_generation_model(self.cfg)
        return GenerationPhase.run(dialogs, model, self.cfg)

    def run_evaluation(self, generated_dialogs: List[Dialog]) -> List[Dict[str, Any]]:
        """Phase 3: Metric evaluation."""
        metrics_map = create_metrics_map(self.dataset, self.cfg)
        return EvaluationPhase.run(generated_dialogs, metrics_map, self.dataset, self.cfg)

    def run_aggregation(self, all_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Phase 4: Result aggregation."""
        return AggregationPhase.run(all_results, self.cfg)

    # ---- Run All ----
    def run(self) -> None:
        """Chain phases based on do_generation / do_evaluation flags in config."""
        generated_dialogs: List[Dialog] = []

        # ── Generation Phase ──
        if self.cfg.do_generation:
            raw_dialogs = self.prepare_data()
            generated_dialogs = self.run_generation(raw_dialogs)

        # ── Evaluation Phase ──
        if self.cfg.do_evaluation:
            # If generation was not run, try loading from disk
            if not generated_dialogs:
                generated_dialogs = self._load_generated_dialogs()
            if not generated_dialogs:
                logger.error("No generated dialogs found. Cannot proceed with evaluation.")
                return

            all_results = self.run_evaluation(generated_dialogs)
            self.run_aggregation(all_results)

    # ---- Helpers ----
    def _load_generated_dialogs(self) -> List[Dialog]:
        """Load previously generated Dialog files from disk."""
        gen_dir = self.cfg.gen_output_dir
        if not gen_dir.exists():
            return []
        logger.info(f"Loading generated dialogs from {gen_dir}")
        dialogs: List[Dialog] = []
        for p in gen_dir.glob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    dialogs.append(Dialog.model_validate_json(f.read()))
            except Exception as e:
                logger.warning(f"Failed to load {p}: {e}")
        dialogs.sort(key=lambda x: x.dialog_id)
        return dialogs
