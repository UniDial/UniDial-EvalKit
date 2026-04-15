"""
Four Phases of the Evaluation Pipeline

  1. DataPhase        — Data preprocessing & loading
  2. GenerationPhase  — Model inference / generation (with checkpoint resume)
  3. EvaluationPhase  — Metric evaluation (with checkpoint resume)
  4. AggregationPhase — Result aggregation & output
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

from src.config import EvalPipelineConfig
from src.dataset.data_utils import to_jsonable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-Dialog processing functions (stateless, concurrency-friendly)
# ---------------------------------------------------------------------------

def _generate_single_dialog(
    dialog: Dialog,
    model: BaseModel,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> Dialog:
    """Generate model responses for a single Dialog, return the updated Dialog."""
    processed_turns = []
    messages: List[Dict[str, str]] = []
    dialog_id = dialog.dialog_id

    model.begin_dialog(dialog_id=dialog_id)
    try:
        for turn in dialog.dialog_turns:
            if turn.role == "system":
                messages.append({"role": "system", "content": turn.content})
                processed_turns.append(turn)

            elif turn.role == "user":
                messages.append({"role": "user", "content": turn.content})
                processed_turns.append(turn)

            elif turn.role == "assistant":

                # Check if this turn requires evaluation; if so, generate; otherwise, keep existing content
                if turn.eval_config.do_eval:
                    gen_res = model.generate(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        dialog_id=dialog_id,
                    )
                    if isinstance(gen_res, tuple) and len(gen_res) == 2:
                        response, response_details = gen_res
                    else:
                        response, response_details = gen_res, None
                        
                    # Build history: use reference or generated response based on dialog config
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
                    # Store raw generation details for later debugging/analysis.
                    # Use `raw_` prefix so downstream evaluation output can filter it out.
                    if response_details is not None:
                        new_turn.turn_labels["raw_response_details"] = to_jsonable(response_details)
                    
                    processed_turns.append(new_turn)
                else:
                    messages.append({"role": "assistant", "content": turn.content})
                    processed_turns.append(turn.model_copy())
    finally:
        # Always cleanup dialog-scoped state for stateful agent models.
        model.end_dialog(dialog_id=dialog_id)

    new_dialog = dialog.model_copy()
    new_dialog.dialog_turns = processed_turns
    return new_dialog


def _evaluate_single_dialog(
    dialog: Dialog,
    metrics_map: Dict[str, Any],
    dataset: BenchmarkDataset,
) -> List[Dict[str, Any]]:
    """Evaluate a single Dialog, return a list of evaluation records."""
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

                    # Build a human-readable metric name
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
# Four Pipeline Phases
# ---------------------------------------------------------------------------

class DataPhase:
    """Phase 1: Data preprocessing & loading"""

    @staticmethod
    def run(dataset: BenchmarkDataset, cfg: EvalPipelineConfig) -> List[Dialog]:
        logger.info("Preprocessing / Loading dataset …")
        processed_path = dataset.preprocess(
            raw_path=cfg.raw_data_dir,
            processed_root=cfg.processed_data_dir,
        )
        dialogs = list(dataset.load_eval_dialogs(data_root=processed_path, recursive=True, require_alternative_roles=cfg.require_alternative_roles))
        logger.info(f"Loaded {len(dialogs)} dialogs.")
        return dialogs


class GenerationPhase:
    """Phase 2: Model generation with checkpoint resume"""

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
    """Phase 3: Metric evaluation with checkpoint resume"""

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
    """Phase 4: Result aggregation & output"""

    @staticmethod
    def run(
        all_results: List[Dict[str, Any]],
        cfg: EvalPipelineConfig,
    ) -> Optional[Dict[str, Any]]:
        # If no results in memory, try loading from disk
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

        # Print summary
        print("\n" + "=" * 40)
        print("Global Results:")
        print(json.dumps(aggregated["global"], indent=2))
        print("=" * 40 + "\n")

        # Persist to disk
        summary_path = cfg.summary_output_path
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"summary": aggregated}, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved summary to {summary_path}")

        return aggregated

