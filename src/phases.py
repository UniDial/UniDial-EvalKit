"""
评测流水线的四个阶段 (Phases)

  1. DataPhase        — 数据预处理与加载
  2. GenerationPhase  — 模型推理生成（支持断点续跑）
  3. EvaluationPhase  — 指标评测（支持断点续跑）
  4. AggregationPhase — 结果聚合与输出
"""

from __future__ import annotations

import json
import logging
import os
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Set

from tqdm import tqdm

from src.dataset import BenchmarkDataset
from src.dataset.schema import Dialog
from src.metric.aggregator import aggregate_results
from src.model import BaseModel

from .eval_config import EvalPipelineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 单 Dialog 处理函数 (无状态, 方便并发)
# ---------------------------------------------------------------------------

def _generate_single_dialog(
    dialog: Dialog,
    model: BaseModel,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> Dialog:
    """为单个 Dialog 生成模型回复, 返回更新后的 Dialog。"""
    processed_turns = []
    messages: List[Dict[str, str]] = []

    for turn in dialog.dialog_turns:
        if turn.role == "system":
            messages.append({"role": "system", "content": turn.content})
            processed_turns.append(turn)

        elif turn.role == "user":
            messages.append({"role": "user", "content": turn.content})
            processed_turns.append(turn)

        elif turn.role == "assistant":

            # 判断本轮是否需要评测，若需要，则generate；若不需要，则加入content
            if turn.eval_config.do_eval:
                response = model.generate(
                    messages=messages, temperature=temperature, max_tokens=max_tokens
                )
                # 构建历史: 根据 dialog 配置决定用参考答案还是生成结果
                if dialog.dialog_eval_config.use_reference_history:
                    messages.append({
                        "role": "assistant",
                        "content": turn.reference if turn.reference is not None else turn.content,
                    })
                else:
                    messages.append({"role": "assistant", "content": response})

                # Create new turn with generated content
                new_turn = turn.model_copy()
                new_turn.content = response
                processed_turns.append(new_turn)
            else:
                messages.append({"role": "assistant", "content": turn.content})
                processed_turns.append(turn.model_copy())

    new_dialog = dialog.model_copy()
    new_dialog.dialog_turns = processed_turns
    return new_dialog


def _evaluate_single_dialog(
    dialog: Dialog,
    metrics_map: Dict[str, Any],
    dataset: BenchmarkDataset,
) -> List[Dict[str, Any]]:
    """对单个 Dialog 执行评测, 返回评测记录列表。"""
    results: List[Dict[str, Any]] = []
    history_messages: List[Dict[str, str]] = []

    for turn in dialog.dialog_turns:
        if turn.role == "user":
            history_messages.append({"role": "user", "content": turn.content})

        elif turn.role == "assistant":
            if turn.eval_config and turn.eval_config.do_eval:

                # Dynamic Config Resolution
                metrics_to_run = dataset.get_eval_config_for_turn(turn)

                for metric_cfg in metrics_to_run:
                    metric_name = metric_cfg.class_name
                    metric_inst = metrics_map.get(metric_name)
                    if not metric_inst:
                        continue

                    score_dict = metric_inst.compute(
                        prediction=turn.content,
                        reference=turn.reference,
                        history_messages=history_messages,
                        dataset=dataset,
                        **metric_cfg.args,
                    )

                    # 构造可读的 metric 名称
                    record_metric_name = metric_name
                    for key, value in metric_cfg.args.items():
                        if "name" in key.lower() and isinstance(value, str):
                            record_metric_name = f"{metric_name}->{value}"
                            break

                    results.append({
                        "dialog_id": dialog.dialog_id,
                        "turn_id": turn.turn_id,
                        "metric_name": record_metric_name,
                        "score": score_dict.get("score", 0.0),
                        "details": score_dict,
                        "dialog_labels": dialog.dialog_labels,
                        "turn_labels": {
                            k: v for k, v in turn.turn_labels.items() if not k.startswith("raw")
                        },
                    })

            history_messages.append({"role": "assistant", "content": turn.content})

    return results


# ---------------------------------------------------------------------------
# 四大阶段 (Phase)
# ---------------------------------------------------------------------------

class DataPhase:
    """阶段 1: 数据预处理 & 加载"""

    @staticmethod
    def run(dataset: BenchmarkDataset, cfg: EvalPipelineConfig) -> List[Dialog]:
        logger.info("Preprocessing / Loading dataset …")
        processed_path = dataset.preprocess(
            raw_path=cfg.raw_data_dir,
            processed_root=cfg.processed_data_dir,
        )
        dialogs = list(dataset.load_eval_dialogs(data_root=processed_path, recursive=True))
        logger.info(f"Loaded {len(dialogs)} dialogs.")
        return dialogs


class GenerationPhase:
    """阶段 2: 模型生成, 支持断点续跑"""

    @staticmethod
    def run(
        dialogs: List[Dialog],
        model: BaseModel,
        cfg: EvalPipelineConfig,
    ) -> List[Dialog]:
        output_dir = cfg.gen_output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 1. Load existing results to support resume
        processed: Dict[int, Dialog] = {}
        existing_files = list(output_dir.glob("*.json"))
        if existing_files:
            logger.info(f"Found {len(existing_files)} existing generated files. Resuming …")
            for p in existing_files:
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        d = Dialog.model_validate_json(f.read())
                        processed[d.dialog_id] = d
                except Exception as e:
                    logger.warning(f"Failed to parse {p}: {e}")

        # 2. Identify remaining tasks
        remaining = [d for d in dialogs if d.dialog_id not in processed]
        if not remaining:
            logger.info("All dialogs already generated. Skipping.")
            return sorted(processed.values(), key=lambda x: x.dialog_id)

        logger.info(f"Generating responses for {len(remaining)} dialogs …")

        # 3. Process remaining dialogs
        with ThreadPoolExecutor(max_workers=cfg.parallel) as executor:
            future_map = {
                executor.submit(
                    _generate_single_dialog, d, model, cfg.temperature, cfg.max_tokens
                ): d
                for d in remaining
            }
            for future in tqdm(
                concurrent.futures.as_completed(future_map),
                total=len(future_map),
                desc="Generating",
            ):
                d = future_map[future]
                try:
                    res = future.result()
                    file_path = output_dir / f"{res.dialog_id}.json"
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(res.model_dump_json(indent=2))
                    processed[res.dialog_id] = res
                except Exception as e:
                    logger.error(f"Generation failed for dialog {d.dialog_id}: {e}")

        return sorted(processed.values(), key=lambda x: x.dialog_id)


class EvaluationPhase:
    """阶段 3: 指标评测, 支持断点续跑"""

    @staticmethod
    def run(
        generated_dialogs: List[Dialog],
        metrics_map: Dict[str, Any],
        dataset: BenchmarkDataset,
        cfg: EvalPipelineConfig,
    ) -> List[Dict[str, Any]]:
        output_dir = cfg.eval_output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 1. Load existing evaluation results
        processed_ids: Set[int] = set()
        all_results: List[Dict[str, Any]] = []
        existing_files = list(output_dir.glob("*.json"))
        if existing_files:
            logger.info(f"Found {len(existing_files)} existing eval files. Resuming …")
            for p in existing_files:
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        # Each file contains a list of result records for that dialog
                        records = json.load(f)
                        if isinstance(records, list):
                            all_results.extend(records)
                            # Assume filename is dialog_id.json or check first record
                            if records:
                                processed_ids.add(records[0]["dialog_id"])
                except Exception as e:
                    logger.warning(f"Failed to parse {p}: {e}")

        remaining = [d for d in generated_dialogs if d.dialog_id not in processed_ids]
        if not remaining:
            logger.info("All dialogs already evaluated.")
            return all_results

        logger.info(f"Evaluating {len(remaining)} dialogs …")

        # 2. Evaluate remaining
        with ThreadPoolExecutor(max_workers=cfg.parallel) as executor:
            future_map = {
                executor.submit(_evaluate_single_dialog, d, metrics_map, dataset): d
                for d in remaining
            }
            for future in tqdm(
                concurrent.futures.as_completed(future_map),
                total=len(future_map),
                desc="Evaluating",
            ):
                d = future_map[future]
                try:
                    res_list = future.result()
                    if res_list:
                        # Write results for this dialog to its own file
                        file_path = output_dir / f"{d.dialog_id}.json"
                        with open(file_path, "w", encoding="utf-8") as f:
                            json.dump(res_list, f, indent=2, ensure_ascii=False)
                        all_results.extend(res_list)
                except Exception as e:
                    logger.error(f"Evaluation failed for dialog {d.dialog_id}: {e}")

        return all_results


class AggregationPhase:
    """阶段 4: 结果聚合 & 输出"""

    @staticmethod
    def run(
        all_results: List[Dict[str, Any]],
        cfg: EvalPipelineConfig,
    ) -> Optional[Dict[str, Any]]:
        # 若内存中无结果, 尝试从磁盘加载
        if not all_results and cfg.eval_output_dir.exists():
            for p in cfg.eval_output_dir.glob("*.json"):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        records = json.load(f)
                        if isinstance(records, list):
                            all_results.extend(records)
                except Exception:
                    pass

        if not all_results:
            logger.warning("No results to aggregate.")
            return None

        logger.info("Aggregating results …")
        aggregated = aggregate_results(
            all_results,
            turn_stat=cfg.agg_turn_stat,
            dialog_stat=cfg.agg_dialog_stat,
            dataset_level=cfg.agg_dataset_level,
            by_metric=cfg.agg_by_metric,
        )

        # 打印摘要
        print("\n" + "=" * 40)
        print("Global Results:")
        print(json.dumps(aggregated["global"], indent=2))
        print("=" * 40 + "\n")

        # 持久化
        summary_path = cfg.summary_output_path
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"summary": aggregated}, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved summary to {summary_path}")

        return aggregated

