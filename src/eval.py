import argparse
import json
import logging
import os
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import glob
import pandas as pd
from tqdm import tqdm

from src.dataset import get_dataset_class, BenchmarkDataset
from src.dataset.schema import Dialog
from src.metric import get_metric_class, METRIC_REGISTRY
from src.metric.aggregator import aggregate_results
from src.model import get_model_class, BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress noisy library logs (HTTP requests etc.)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation Pipeline")
    parser.add_argument("--dataset", type=str, default="mt_eval", help="Benchmark dataset name (e.g., mt_eval)")
    parser.add_argument("--raw_data_dir", type=str, default="./raw_data/MT-Eval", help="Path to raw dataset files")
    parser.add_argument("--processed_data_dir", type=str, default="./data", help="Path to store/load processed dialogs")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save evaluation results")
    parser.add_argument("--model_type", type=str, default="openai", help="Type of model to use (openai, etc.)")
    parser.add_argument("--model_name", type=str, default="deepseek-chat", help="Model to evaluate (e.g., gpt-3.5-turbo)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for model generation")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum tokens for model generation")
    parser.add_argument("--judge_model_type", type=str, default="openai", help="Type of judge model to use (openai, etc.)")
    parser.add_argument("--judge_model_name", type=str, default="deepseek-chat", help="Judge model name for LLM-based metrics")
    parser.add_argument("--parallel", type=int, default=4, help="Number of parallel threads/processes")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key (or set OPENAI_API_KEY)")
    parser.add_argument("--base_url", type=str, default=None, help="OpenAI API Base URL")
    
    # Task control flags
    parser.add_argument("--do_generation", action="store_true", help="Run the generation phase")
    parser.add_argument("--do_evaluation", action="store_true", help="Run the evaluation phase")
    
    # Aggregation argument
    parser.add_argument(
        "--agg_by_metric",
        action="store_true",
        help="Whether to aggregate results by metric name",
    )
    parser.add_argument(
        "--agg_turn_stat",
        dest="agg_turn_stat",
        type=str,
        default="mean",
        choices=["mean", "min", "max"],
        help="Aggregation statistic for combining multiple metric scores within a single turn",
    )
    parser.add_argument(
        "--agg_dialog_stat",
        dest="agg_dialog_stat",
        type=str,
        default="min",
        choices=["mean", "min", "max"],
        help="Aggregation statistic for combining turn scores within a dialog",
    )
    parser.add_argument(
        "--agg_dataset_level",
        dest="agg_dataset_level",
        type=str,
        default="dialog",
        choices=["dialog", "turn"],
        help='Aggregation level across the dataset: "dialog" (avg dialog scores) or "turn" (avg all turns flattened)',
    )
    
    
    
    return parser.parse_args()

def process_single_dialog_generation(
    dialog: Dialog, 
    model: BaseModel,
    temperature: float = 0.7,
    max_tokens: int = 1024
) -> Dialog:
    """
    Generate model responses for a single dialog.
    Updates the assistant turns with generated content.
    """
    processed_turns = []
    messages = []
    
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
                # Generate response
                try:
                    response = model.generate(messages=messages, temperature=temperature, max_tokens=max_tokens)
                except Exception as e:
                    logger.error(f"Generation failed for dialog {dialog.dialog_id}, turn {turn.turn_id}: {e}")
                    response = "[ERROR]"
                
                # Update history with generated response
                if dialog.dialog_eval_config.use_reference_history:                    
                    messages.append({"role": "assistant", "content": turn.reference if turn.reference is not None else turn.content})
                else:     
                    messages.append({"role": "assistant", "content": response})
                
                # Create new turn with generated content
                new_turn = turn.model_copy()
                new_turn.content = response
                processed_turns.append(new_turn)
            else:
                messages.append({"role": "assistant", "content": turn.content})
                new_turn = turn.model_copy()
                processed_turns.append(new_turn)
            
    new_dialog = dialog.model_copy()
    new_dialog.dialog_turns = processed_turns
    return new_dialog

def run_generation_phase(
    dialogs: List[Dialog],
    model: BaseModel,
    output_dir: Path,
    parallel: int = 4,
    temperature: float = 0.7,
    max_tokens: int = 1024
) -> List[Dialog]:
    """
    Run the generation phase with file-per-dialog checkpointing.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load existing results to support resume
    processed_dialogs = {}
    existing_files = list(output_dir.glob("*.json"))
    
    if existing_files:
        logger.info(f"Found {len(existing_files)} existing generated files in {output_dir}. Resuming...")
        for p in existing_files:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    d = Dialog.model_validate_json(f.read())
                    processed_dialogs[d.dialog_id] = d
            except Exception as e:
                logger.warning(f"Failed to parse generation file {p}: {e}")
    
    # 2. Identify remaining tasks
    remaining_dialogs = [d for d in dialogs if d.dialog_id not in processed_dialogs]
    
    if not remaining_dialogs:
        logger.info("All dialogs already generated. Skipping generation phase.")
        return sorted(processed_dialogs.values(), key=lambda x: x.dialog_id)
    
    logger.info(f"Generating responses for {len(remaining_dialogs)} dialogs...")
    
    # 3. Process remaining dialogs
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        future_to_dialog = {
            executor.submit(process_single_dialog_generation, d, model, temperature, max_tokens): d 
            for d in remaining_dialogs
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_dialog), total=len(future_to_dialog), desc="Generating"):
            d = future_to_dialog[future]
            try:
                res_dialog = future.result()
                
                # Write to individual file
                file_path = output_dir / f"{res_dialog.dialog_id}.json"
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(res_dialog.model_dump_json(indent=2))
                
                processed_dialogs[res_dialog.dialog_id] = res_dialog
                
            except Exception as e:
                logger.error(f"Generation generated an exception for dialog {d.dialog_id}: {e}")

    return sorted(processed_dialogs.values(), key=lambda x: x.dialog_id)


def process_single_dialog_evaluation(
    dialog: Dialog, 
    metrics_map: Dict[str, Any], 
    dataset: BenchmarkDataset
) -> List[Dict[str, Any]]:
    """
    Evaluate a single dialog using initialized metrics.
    """
    results = []
    history_messages = []
    
    for turn in dialog.dialog_turns:
        if turn.role == "user":
            history_messages.append({"role": "user", "content": turn.content})
        elif turn.role == "assistant":
            if turn.eval_config and turn.eval_config.do_eval:
                for metric_cfg in turn.eval_config.metrics:
                    metric_name = metric_cfg.class_name
                    metric_inst = metrics_map.get(metric_name)
                    
                    if not metric_inst:
                        continue
                        
                    try:
                        score_dict = metric_inst.compute(
                            prediction=turn.content,
                            reference=turn.reference, 
                            history_messages=history_messages,
                            dataset=dataset, 
                            **metric_cfg.args 
                        )
                        
                        record_metric_name = metric_name
                        for key, value in metric_cfg.args.items():
                            if "name" in key.lower() and isinstance(value, str):
                                record_metric_name = f"{metric_name}->{value}"
                                break

                        result_record = {
                            "dialog_id": dialog.dialog_id,
                            "turn_id": turn.turn_id,
                            "metric_name": record_metric_name,
                            "score": score_dict.get("score", 0.0),
                            "details": score_dict,
                            # Keep both separately for analysis/debugging.
                            "dialog_labels": dialog.dialog_labels,
                            "turn_labels": {k: v for k, v in turn.turn_labels.items() if not k.startswith("raw")},
                        }
                        results.append(result_record)
                        
                    except Exception as e:
                        logger.error(f"Evaluation failed for dialog {dialog.dialog_id}: {e}")
            
            history_messages.append({"role": "assistant", "content": turn.content})
            
    return results

def run_evaluation_phase(
    generated_dialogs: List[Dialog],
    metrics_map: Dict[str, Any],
    dataset: BenchmarkDataset,
    output_dir: Path,
    parallel: int = 4
) -> List[Dict[str, Any]]:
    """
    Run the evaluation phase with file-per-dialog checkpointing.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load existing evaluation results
    processed_dialog_ids: Set[int] = set()
    all_results: List[Dict[str, Any]] = []
    existing_files = list(output_dir.glob("*.json"))
    
    if existing_files:
        logger.info(f"Found {len(existing_files)} existing eval result files in {output_dir}. Resuming...")
        for p in existing_files:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    # Each file contains a list of result records for that dialog
                    records = json.load(f)
                    if isinstance(records, list):
                        all_results.extend(records)
                        # Assume filename is dialog_id.json or check first record
                        if records:
                             processed_dialog_ids.add(records[0]["dialog_id"])
            except Exception as e:
                logger.warning(f"Failed to parse eval file {p}: {e}")
            
    remaining_dialogs = [d for d in generated_dialogs if d.dialog_id not in processed_dialog_ids]
    
    if not remaining_dialogs:
        logger.info("All dialogs already evaluated.")
        return all_results

    logger.info(f"Evaluating {len(remaining_dialogs)} dialogs...")
    
    # 2. Evaluate remaining
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        future_to_dialog = {
            executor.submit(process_single_dialog_evaluation, d, metrics_map, dataset): d
            for d in remaining_dialogs
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_dialog), total=len(future_to_dialog), desc="Evaluating"):
            d = future_to_dialog[future]
            try:
                res_list = future.result()
                if res_list:
                    # Write results for this dialog to its own file
                    file_path = output_dir / f"{d.dialog_id}.json"
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(res_list, f, indent=2, ensure_ascii=False)
                    
                    all_results.extend(res_list)
                else:
                    # Even if empty results (no eval needed?), mark as processed?
                    # Maybe create empty file to avoid re-processing?
                    pass
                    
            except Exception as e:
                logger.error(f"Evaluation generated an exception for dialog {d.dialog_id}: {e}")
                    
    return all_results

def main():
    args = parse_args()
    
    # Use directories instead of single files
    gen_output_dir = Path(args.output_dir) / f"{args.dataset}" / f"{args.model_name}" / "generated"
    eval_output_dir = Path(args.output_dir) / f"{args.dataset}" / f"{args.model_name}" / "eval_details"
    
    logger.info(f"Initialize dataset: {args.dataset}")
    DatasetClass = get_dataset_class(args.dataset)
    dataset = DatasetClass()
    
    # --- 1. Generation Phase ---
    generated_dialogs = []
    
    if args.do_generation:
        logger.info(f"Starting generation phase...")
        
        # Preprocess and load
        # try:
        logger.info("Preprocessing/Loading dataset...")
        processed_path = dataset.preprocess(
            raw_path=args.raw_data_dir, 
            processed_root=args.processed_data_dir
        )
        raw_dialogs = list(dataset.load_eval_dialogs(data_root=processed_path, recursive=True))
        # except FileNotFoundError as e:
        #     logger.error(f"Data processing failed. {e}")
        #     # logger.error(f"Check if raw_data_dir exists: {args.raw_data_dir}")
        #     # logger.error(f"Current working directory: {os.getcwd()}")
        #     return
        # except Exception as e:
        #     logger.error(f"Dataset loading failed: {e}")
        #     return

        # Initialize Generation Model
        logger.info(f"Initializing model: {args.model_name} (Type: {args.model_type})")
        
        ModelClass = get_model_class(args.model_type)
        gen_model = ModelClass( 
            model_name=args.model_name, 
            api_key=args.api_key, 
            base_url=args.base_url,
        )# TODO: 可能agent还需要load其他embedding model
        
        # Run Generation
        generated_dialogs = run_generation_phase(
            dialogs=raw_dialogs,
            model=gen_model,
            output_dir=gen_output_dir,
            parallel=args.parallel,
            temperature=args.temperature
        )
    
    # --- 2. Evaluation Phase ---
    all_results = []
    
    if args.do_evaluation:
        logger.info(f"Starting evaluation phase...")
        
        # Load generated dialogs if not already loaded
        if not generated_dialogs and gen_output_dir.exists():
            logger.info(f"Loading generated dialogs from {gen_output_dir}")
            try:
                # Load all json files from the directory
                for p in gen_output_dir.glob("*.json"):
                    with open(p, "r", encoding="utf-8") as f:
                        generated_dialogs.append(Dialog.model_validate_json(f.read()))
                # Sort to ensure consistent order (though not strictly necessary for correctness)
                generated_dialogs.sort(key=lambda x: x.dialog_id)
            except Exception as e:
                logger.error(f"Failed to load generated dialogs: {e}")
                return

        if not generated_dialogs:
            logger.error("No generated dialogs found. Cannot proceed with evaluation.")
            return


        logger.info("Starting evaluation phase...")
        

        metrics_map = {}
        metric_configs = dataset.metric_configs()
        
        # Normalize config to dict if it's a list
        if isinstance(metric_configs, list):
            metric_configs = {name: {} for name in metric_configs}
            
        for name, config in metric_configs.items():
            # print(name, config)
            if name not in METRIC_REGISTRY:
                logger.warning(f"Metric {name} not found in registry. Skipping.")
                continue
                
            if name == "llm_judge":
                JudgeModelClass = get_model_class("openai") 
                judge_model = JudgeModelClass(
                    model_name=args.judge_model_name,
                    api_key=args.api_key,
                    base_url=args.base_url
                )
        
                metrics_map[name] = get_metric_class(name)(llm_client=judge_model, dataset=dataset, **config)
            else:
                metrics_map[name] = get_metric_class(name)(**config)
        
        
        all_results = run_evaluation_phase(
            generated_dialogs=generated_dialogs,
            metrics_map=metrics_map,
            dataset=dataset,
            output_dir=eval_output_dir,
            parallel=args.parallel
        )
        
        # --- 3. Aggregation Phase ---
        logger.info("Aggregating results...")
        
        # If all_results is empty (e.g. all skipped), load from file to aggregate
        if not all_results and eval_output_dir.exists():
             for p in eval_output_dir.glob("*.json"):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        records = json.load(f)
                        if isinstance(records, list):
                            all_results.extend(records)
                except Exception:
                    pass
        
        if all_results:
            aggregated = aggregate_results(
                all_results, 
                turn_stat=args.agg_turn_stat,
                dialog_stat=args.agg_dialog_stat,
                dataset_level=args.agg_dataset_level,
                by_metric=args.agg_by_metric
            )
            
            # Print Summary
            print("\n" + "="*40)
            print("Global Results:")
            print(json.dumps(aggregated["global"], indent=2))
            print("="*40 + "\n")

            # Save final aggregated summary
            summary_output_path = Path(args.output_dir) / f"{args.dataset}" / f"{args.model_name}" / "summary.json"
            with open(summary_output_path, "w", encoding="utf-8") as f:
                json.dump({
                    "summary": aggregated,
                    # "details": all_results # Too big to dump here usually
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved summary to {summary_output_path}")
        else:
            logger.warning("No results to aggregate.")

if __name__ == "__main__":
    main()
