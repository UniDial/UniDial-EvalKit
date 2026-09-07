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
from src.dataset.schema import Dialog, Turn, TurnEvalConfig
from src.metric.aggregator import aggregate_results
from src.model import BaseModel

from src.config import EvalPipelineConfig
from src.dataset.data_utils import to_jsonable

logger = logging.getLogger(__name__)


def _resolve_user_simulator_calls(
    turn: Turn,
    dataset: BenchmarkDataset,
    cfg_override: Optional[int],
) -> int:
    raw = turn.eval_config.user_simulator_calls
    if raw == 0:
        return 0
    if raw > 0:
        return raw
    # raw == -1 (or any negative): use config override or dataset default
    if cfg_override is not None:
        return max(int(cfg_override), 0)
    return max(int(dataset.get_default_user_simulator_calls(turn)), 0)


def _append_history(
    messages: List[Dict[str, str]],
    turn: Turn,
    *,
    content: Optional[str] = None,
    already_appended: bool = False,
) -> None:
    """Append a turn according to its history-retention label."""
    include_in_history = turn.turn_labels.get("raw_include_in_history", True) is not False

    if already_appended:
        if not messages or messages[-1].get("role") != turn.role:
            raise ValueError(f"Generation history does not end with a {turn.role} turn")
        if not include_in_history:
            messages.pop()
        return

    if not include_in_history:
        return
    messages.append({"role": turn.role, "content": content if content is not None else turn.content})


def _call_model_generate(
    model: BaseModel,
    messages: List[Dict[str, str]],
    *,
    temperature: Optional[float],
    max_tokens: Optional[int],
    dialog_id: Any,
) -> tuple:
    gen_res = model.generate(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        dialog_id=dialog_id,
    )
    if isinstance(gen_res, tuple) and len(gen_res) == 2:
        return gen_res
    return gen_res, None


# ---------------------------------------------------------------------------
# Single-Dialog processing functions (stateless, concurrency-friendly)
# ---------------------------------------------------------------------------

def _generate_single_dialog(
    dialog: Dialog,
    model: BaseModel,
    dataset: BenchmarkDataset,
    user_sim_model: Optional[BaseModel] = None,
    temperature: Optional[float] = None,
    max_tokens: int = 1024,
    max_user_simulator_calls_override: Optional[int] = None,
    checkpoint_path: Optional[Any] = None,
    resume_dialog: Optional[Dialog] = None,
) -> Dialog:
    """Generate one dialog, checkpoint each new model output, and resume saved turns."""
    processed_turns: List[Turn] = []
    messages: List[Dict[str, str]] = []
    dialog_id = dialog.dialog_id

    # Index completed predefined turns by turn_id and simulator turns by their
    # anchor user turn so both kinds can be replayed without another model call.
    saved_turns = resume_dialog.dialog_turns if resume_dialog is not None else []
    saved_predefined = {
        turn.turn_id: turn
        for turn in saved_turns
        if not turn.turn_labels.get("synthesized", False)
    }
    saved_synthesized: Dict[int, List[Turn]] = {}
    for turn in saved_turns:
        if turn.turn_labels.get("synthesized", False):
            anchor_id = turn.turn_labels.get("simulator_anchor_turn_id")
            saved_synthesized.setdefault(anchor_id, []).append(turn)

    # Continue synthesized IDs after both original and previously saved turns.
    next_turn_id = max(
        (t.turn_id for t in [*dialog.dialog_turns, *saved_turns]),
        default=-1,
    ) + 1

    def save_checkpoint(complete: bool = False) -> Dialog:
        result = dialog.model_copy(deep=True)
        result.dialog_turns = list(processed_turns)
        if complete:
            result.dialog_labels.pop("_generation_incomplete", None)
        else:
            result.dialog_labels["_generation_incomplete"] = True
        if checkpoint_path is not None:
            # Keep the previous valid checkpoint until the new JSON is complete.
            tmp_path = f"{checkpoint_path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(result.model_dump_json(indent=2))
            os.replace(tmp_path, checkpoint_path)
        return result

    model.begin_dialog(dialog_id=dialog_id)
    try:
        for turn_index, turn in enumerate(dialog.dialog_turns):
            if turn.role == "system":
                _append_history(messages, turn)
                processed_turns.append(turn)

            elif turn.role == "user":
                # Always expose the user input to the upcoming generation.
                # Durable retention is decided after the assistant turn.
                messages.append({"role": "user", "content": turn.content})
                processed_turns.append(turn)

            elif turn.role == "assistant":
                preceding_user = (
                    dialog.dialog_turns[turn_index - 1]
                    if turn_index > 0 and dialog.dialog_turns[turn_index - 1].role == "user"
                    else None
                )
                should_generate = bool(turn.eval_config.do_eval) or turn.content is None

                # A saved turn is replayed into history; only a missing turn is generated.
                saved_turn = saved_predefined.get(turn.turn_id)
                new_turn = (
                    saved_turn.model_copy(deep=True)
                    if saved_turn is not None
                    else turn.model_copy(deep=True)
                )

                if should_generate and saved_turn is None:
                    response, response_details = _call_model_generate(
                        model,
                        messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        dialog_id=dialog_id,
                    )
                    new_turn.content = response
                    if response_details is not None:
                        new_turn.turn_labels["raw_response_details"] = to_jsonable(response_details)

                if dialog.dialog_eval_config.use_reference_history and turn.reference is not None:
                    hist_content = turn.reference if isinstance(turn.reference, str) else str(turn.reference)
                else:
                    hist_content = new_turn.content

                if preceding_user is not None:
                    _append_history(messages, preceding_user, already_appended=True)
                _append_history(messages, new_turn, content=hist_content)
                processed_turns.append(new_turn)
                if should_generate and saved_turn is None:
                    save_checkpoint()

                # After each predefined user→assistant pair, optionally run simulator.
                if (
                    dialog.dialog_eval_config.enable_user_simulator
                    and preceding_user is not None
                    and user_sim_model is not None
                ):
                    n_calls = _resolve_user_simulator_calls(
                        preceding_user, dataset, max_user_simulator_calls_override
                    )
                    # Resume at any point in the saved simulator user/assistant sequence.
                    resumed_sim_turns = saved_synthesized.get(preceding_user.turn_id, [])
                    resumed_sim_index = 0
                    for _ in range(n_calls):
                        generated_sim_user = False
                        if (
                            resumed_sim_index < len(resumed_sim_turns)
                            and resumed_sim_turns[resumed_sim_index].role == "user"
                        ):
                            sim_user_turn = resumed_sim_turns[resumed_sim_index].model_copy(deep=True)
                            resumed_sim_index += 1
                        else:
                            prompt = dataset.render_user_simulator_prompt(
                                dialog=dialog,
                                history_messages=messages,
                                current_turn=preceding_user,
                            )
                            sim_user_text, sim_user_details = _call_model_generate(
                                user_sim_model,
                                [{"role": "user", "content": prompt}],
                                temperature=temperature,
                                max_tokens=max_tokens,
                                dialog_id=dialog_id,
                            )
                            sim_user_turn = Turn(
                                turn_id=next_turn_id,
                                role="user",
                                content=sim_user_text,
                                eval_config=TurnEvalConfig(do_eval=False, user_simulator_calls=0),
                                turn_labels={
                                    "synthesized": True,
                                    "simulator_anchor_turn_id": preceding_user.turn_id,
                                    **(
                                        {"raw_response_details": to_jsonable(sim_user_details)}
                                        if sim_user_details is not None
                                        else {}
                                    ),
                                },
                            )
                            next_turn_id += 1
                            generated_sim_user = True
                        _append_history(messages, sim_user_turn)
                        processed_turns.append(sim_user_turn)
                        if generated_sim_user:
                            save_checkpoint()

                        generated_sim_assistant = False
                        if (
                            resumed_sim_index < len(resumed_sim_turns)
                            and resumed_sim_turns[resumed_sim_index].role == "assistant"
                        ):
                            sim_asst_turn = resumed_sim_turns[resumed_sim_index].model_copy(deep=True)
                            resumed_sim_index += 1
                        else:
                            sim_asst_text, sim_asst_details = _call_model_generate(
                                model,
                                messages,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                dialog_id=dialog_id,
                            )
                            sim_asst_turn = Turn(
                                turn_id=next_turn_id,
                                role="assistant",
                                content=sim_asst_text,
                                eval_config=TurnEvalConfig(do_eval=False, user_simulator_calls=0),
                                turn_labels={
                                    "synthesized": True,
                                    "simulator_anchor_turn_id": preceding_user.turn_id,
                                    **(
                                        {"raw_response_details": to_jsonable(sim_asst_details)}
                                        if sim_asst_details is not None
                                        else {}
                                    ),
                                },
                            )
                            next_turn_id += 1
                            generated_sim_assistant = True
                        _append_history(messages, sim_asst_turn)
                        processed_turns.append(sim_asst_turn)
                        if generated_sim_assistant:
                            save_checkpoint()
    finally:
        # Always cleanup dialog-scoped state for stateful agent models.
        model.end_dialog(dialog_id=dialog_id)

    return save_checkpoint(complete=True)


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
            if turn.turn_labels.get("raw_include_in_history", True) is not False:
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
                    
                    if "\n\nAnswer:" in turn.content:
                        content = turn.content.split("\n\nAnswer:")[1].strip()
                    else:
                        content = turn.content
                    score_dict = metric_inst.compute(
                        prediction=content, #turn.content,
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

            if turn.turn_labels.get("raw_include_in_history", True) is not False:
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
        if cfg.raw_data_dir:
            processed_path = dataset.preprocess(
                raw_path=cfg.raw_data_dir,
                processed_root=cfg.processed_data_dir,
            )
        else:
            processed_path = os.path.join(cfg.processed_data_dir, dataset.dataset_name)
            meta_path = os.path.join(processed_path, "_meta.json")
            if not os.path.isdir(processed_path) or not os.path.isfile(meta_path):
                raise FileNotFoundError(
                    f"Processed dataset not found at {processed_path}; "
                    "provide --raw_data_dir to preprocess it"
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
        dataset: Optional[BenchmarkDataset] = None,
        user_sim_model: Optional[BaseModel] = None,
    ) -> List[Dialog]:
        output_dir = cfg.gen_output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 0. Load kept dialog ids # modified
        kept_dialog_ids_path = os.path.join(cfg.down_sampling_dir, f"{cfg.dataset}.json")
        if os.path.exists(kept_dialog_ids_path):
            kept_dialog_ids = json.load(open(kept_dialog_ids_path, "r", encoding="utf-8"))
        else:
            kept_dialog_ids = None

        # 1. Load existing results to support resume
        processed: Dict[int, Dialog] = {}
        partial: Dict[int, Dialog] = {}
        existing_files = list(output_dir.glob("*.json"))
        if existing_files:
            logger.info(f"Found {len(existing_files)} existing generated files. Resuming …")
            for p in existing_files:
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        d = Dialog.model_validate_json(f.read())
                        if d.dialog_labels.get("_generation_incomplete"):
                            partial[d.dialog_id] = d
                        else:
                            processed[d.dialog_id] = d
                except Exception as e:
                    logger.warning(f"Failed to parse {p}: {e}")

        # 2. Identify remaining tasks
        remaining = [d for d in dialogs if d.dialog_id not in processed and (kept_dialog_ids is None or int(d.dialog_id) in kept_dialog_ids)] # modified
        
        if not remaining:
            logger.info("All dialogs already generated. Skipping.")
            return sorted(processed.values(), key=lambda x: x.dialog_id)

        if dataset is None:
            raise ValueError("GenerationPhase.run requires dataset for user-simulator / prompt hooks")

        logger.info(f"Generating responses for {len(remaining)} dialogs …")

        # 3. Process remaining dialogs
        with ThreadPoolExecutor(max_workers=cfg.parallel) as executor:
            future_map = {
                executor.submit(
                    _generate_single_dialog,
                    d,
                    model,
                    dataset,
                    user_sim_model,
                    cfg.temperature,
                    cfg.max_tokens,
                    cfg.max_user_simulator_calls,
                    output_dir / f"{d.dialog_id}.json",
                    partial.get(d.dialog_id),
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
        
        # 0. Load kept dialog ids # modified
        kept_dialog_ids_path = os.path.join(cfg.down_sampling_dir, f"{cfg.dataset}.json")
        if os.path.exists(kept_dialog_ids_path):
            kept_dialog_ids = json.load(open(kept_dialog_ids_path, "r", encoding="utf-8"))
        else:
            kept_dialog_ids = None

        # 1. Load existing evaluation results
        processed_ids: Set[int] = set()
        all_results: List[Dict[str, Any]] = []
        existing_files = list(output_dir.glob("*.json"))
        if existing_files:
            logger.info(f"Found {len(existing_files)} existing eval files. Resuming …")
            for p in existing_files:

                # modified
                if kept_dialog_ids is not None and int(p.stem) not in kept_dialog_ids:
                    continue
                
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

        remaining = [d for d in generated_dialogs if d.dialog_id not in processed_ids and (kept_dialog_ids is None or int(d.dialog_id) in kept_dialog_ids)] # modified
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

