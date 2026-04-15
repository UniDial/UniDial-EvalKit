"""
CLI entry point: parse command-line arguments and invoke EvalPipeline to run the evaluation pipeline.

Usage:
    python src/eval_cli.py --dataset mt_eval --do_generation --do_evaluation
"""

import argparse
import logging

from src.eval_pipeline import EvalPipeline, EvalPipelineConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation Pipeline")
    parser.add_argument("--dataset", type=str, default="mt_eval", help="Benchmark dataset name (e.g., mt_eval)")
    parser.add_argument("--raw_data_dir", type=str, default="./raw_data/MT-Eval", help="Path to raw dataset files")
    parser.add_argument("--processed_data_dir", type=str, default="./data", help="Path to store/load processed dialogs")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save evaluation results")
    parser.add_argument("--require_alternative_roles", action="store_true", help="Whether to require alternative roles, depending on the chat template")
    parser.add_argument("--model_type", type=str, default="openai", help="Type of model to use (openai, etc.)")
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-V3.2", help="Model to evaluate (e.g., gpt-3.5-turbo)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for model generation")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum tokens for model generation")
    parser.add_argument("--judge_model_type", type=str, default="openai", help="Type of judge model to use (openai, etc.)")
    parser.add_argument("--judge_model_name", type=str, default="gpt-4.1-2025-04-14", help="Judge model name for LLM-based metrics")
    parser.add_argument("--parallel", type=int, default=4, help="Number of parallel threads/processes")
    parser.add_argument("--api_key", type=str, default=None, help="API key (or set OPENAI_API_KEY)")
    parser.add_argument("--base_url", type=str, default=None, help="API Base URL")
    parser.add_argument("--embedding_model_name", type=str, default="text-embedding-ada-002", help="Embedding model to use for RAG")
    parser.add_argument("--save_agent_logs", type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=True, help="Whether to save detailed diagnostic logs for agents (default: True)")

    # Task control flags
    parser.add_argument("--do_generation", action="store_true", help="Run the generation phase")
    parser.add_argument("--do_evaluation", action="store_true", help="Run the evaluation phase")

    # Aggregation arguments
    parser.add_argument("--agg_by_metric", action="store_true", help="Whether to aggregate results by metric name")
    parser.add_argument("--agg_turn_stat", type=str, default="mean", choices=["mean", "min", "max"],
                        help="Aggregation statistic for combining multiple metric scores within a single turn")
    parser.add_argument("--agg_dialog_stat", type=str, default="min", choices=["mean", "min", "max"],
                        help="Aggregation statistic for combining turn scores within a dialog")
    parser.add_argument("--agg_dataset_level", type=str, default="dialog", choices=["dialog", "turn"],
                        help='Aggregation level across the dataset: "dialog" (avg dialog scores) or "turn" (avg all turns flattened)')

    return parser.parse_args()


def args_to_config(args: argparse.Namespace) -> EvalPipelineConfig:
    """Convert argparse Namespace to EvalPipelineConfig."""
    return EvalPipelineConfig(
        dataset=args.dataset,
        raw_data_dir=args.raw_data_dir,
        processed_data_dir=args.processed_data_dir,
        output_dir=args.output_dir,
        require_alternative_roles=args.require_alternative_roles,
        model_type=args.model_type,
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        judge_model_type=args.judge_model_type,
        judge_model_name=args.judge_model_name,
        api_key=args.api_key,
        base_url=args.base_url,
        parallel=args.parallel,
        do_generation=args.do_generation,
        do_evaluation=args.do_evaluation,
        save_agent_logs=args.save_agent_logs,
        agg_by_metric=args.agg_by_metric,
        agg_turn_stat=args.agg_turn_stat,
        agg_dialog_stat=args.agg_dialog_stat,
        agg_dataset_level=args.agg_dataset_level,
    )


def main():
    args = parse_args()
    cfg = args_to_config(args)
    pipeline = EvalPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()

