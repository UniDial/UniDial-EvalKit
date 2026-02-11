"""
模块化评测流水线 (Evaluation Pipeline)

将评测流程拆分为四个独立阶段:
  1. DataPhase        — 数据预处理与加载
  2. GenerationPhase  — 模型推理生成
  3. EvaluationPhase  — 指标评测
  4. AggregationPhase — 结果聚合

每个阶段可独立调用, 也可通过 EvalPipeline.run() 一键串联执行。

本文件是统一入口, 外部使用只需:
    from src.eval_pipeline import EvalPipeline, EvalPipelineConfig
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.dataset import get_dataset_class, BenchmarkDataset
from src.dataset.schema import Dialog
from src.metric import get_metric_class, METRIC_REGISTRY
from src.model import get_model_class, BaseModel

# Re-export: 配置 & 阶段
from .eval_config import EvalPipelineConfig
from .eval_phases import (
    DataPhase,
    GenerationPhase,
    EvaluationPhase,
    AggregationPhase,
)

logger = logging.getLogger(__name__)

__all__ = [
    # 配置
    "EvalPipelineConfig",
    # 阶段
    "DataPhase",
    "GenerationPhase",
    "EvaluationPhase",
    "AggregationPhase",
    # 工厂
    "create_generation_model",
    "create_metrics_map",
    # 主控
    "EvalPipeline",
]


# ---------------------------------------------------------------------------
# 模型 & 指标工厂 (集中管理初始化逻辑)
# ---------------------------------------------------------------------------

def create_generation_model(cfg: EvalPipelineConfig) -> BaseModel:
    """根据配置创建生成模型实例。"""
    logger.info(f"Initializing generation model: {cfg.model_name} (type={cfg.model_type})")
    ModelClass = get_model_class(cfg.model_type)
    return ModelClass(
        model_name=cfg.model_name,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
    )


def create_metrics_map(
    dataset: BenchmarkDataset,
    cfg: EvalPipelineConfig,
) -> Dict[str, Any]:
    """根据数据集配置和评测参数, 初始化所有所需的 Metric 实例。"""
    metric_configs = dataset.metric_configs()

    # 兼容 list / dict 两种返回格式
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
            )
            metrics_map[name] = get_metric_class(name)(
                llm_client=judge_model, dataset=dataset, **config
            )
        else:
            metrics_map[name] = get_metric_class(name)(**config)

    return metrics_map


# ---------------------------------------------------------------------------
# Pipeline 主控类
# ---------------------------------------------------------------------------

class EvalPipeline:
    """
    评测流水线主控类。

    用法 1 — 一键执行:
        pipeline = EvalPipeline(cfg)
        pipeline.run()

    用法 2 — 按阶段调用:
        pipeline = EvalPipeline(cfg)
        dialogs = pipeline.prepare_data()
        generated = pipeline.run_generation(dialogs)
        results  = pipeline.run_evaluation(generated)
        summary  = pipeline.run_aggregation(results)
    """

    def __init__(self, cfg: EvalPipelineConfig) -> None:
        self.cfg = cfg
        self.dataset: BenchmarkDataset = self._init_dataset()

    # ---- 初始化 ----
    def _init_dataset(self) -> BenchmarkDataset:
        logger.info(f"Initialize dataset: {self.cfg.dataset}")
        DatasetClass = get_dataset_class(self.cfg.dataset)
        return DatasetClass()

    # ---- 阶段方法 ----
    def prepare_data(self) -> List[Dialog]:
        """阶段 1: 数据预处理 & 加载。"""
        return DataPhase.run(self.dataset, self.cfg)

    def run_generation(self, dialogs: List[Dialog]) -> List[Dialog]:
        """阶段 2: 模型生成。"""
        model = create_generation_model(self.cfg)
        return GenerationPhase.run(dialogs, model, self.cfg)

    def run_evaluation(self, generated_dialogs: List[Dialog]) -> List[Dict[str, Any]]:
        """阶段 3: 指标评测。"""
        metrics_map = create_metrics_map(self.dataset, self.cfg)
        return EvaluationPhase.run(generated_dialogs, metrics_map, self.dataset, self.cfg)

    def run_aggregation(self, all_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """阶段 4: 结果聚合。"""
        return AggregationPhase.run(all_results, self.cfg)

    # ---- 一键执行 ----
    def run(self) -> None:
        """根据配置中的 do_generation / do_evaluation 标志串联执行。"""
        generated_dialogs: List[Dialog] = []

        # ── 生成阶段 ──
        if self.cfg.do_generation:
            raw_dialogs = self.prepare_data()
            generated_dialogs = self.run_generation(raw_dialogs)

        # ── 评测阶段 ──
        if self.cfg.do_evaluation:
            # 若生成阶段未执行, 尝试从磁盘加载
            if not generated_dialogs:
                generated_dialogs = self._load_generated_dialogs()
            if not generated_dialogs:
                logger.error("No generated dialogs found. Cannot proceed with evaluation.")
                return

            all_results = self.run_evaluation(generated_dialogs)
            self.run_aggregation(all_results)

    # ---- 辅助 ----
    def _load_generated_dialogs(self) -> List[Dialog]:
        """从磁盘加载已生成的 Dialog 文件。"""
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
