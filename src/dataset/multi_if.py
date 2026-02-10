from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Any

from .base import BenchmarkContext, BenchmarkDataset
from .schema import Dialog, Turn, TurnEvalConfig, MetricConfig, DialogEvalConfig

logger = logging.getLogger(__name__)

class MultiIFDataset(BenchmarkDataset):
    benchmark_id = "multi_if"

    def metric_configs(self) -> List[str]:
        return ["instruction_following"]

    def _normalize_raw_data(self, ctx: BenchmarkContext) -> Iterable[Dialog]:
        raw_path = ctx.raw_path
        logger.info(f"MultiIFDataset: checking raw_path: {raw_path.absolute()}")
        if not raw_path.exists():
            logger.error(f"MultiIFDataset: raw_path does not exist: {raw_path}")
            raise FileNotFoundError(raw_path)
            
        # Look for the csv file
        file_path = raw_path / "multiIF_20241018.csv"
        with file_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            for line_index, row in enumerate(reader):
                
                # if line_index == 5:
                #     break
                
                yield self._build_dialog(
                    row,
                    source_file=file_path.name,
                    line_index=line_index,
                )

    def _build_dialog(
        self,
        row: Dict[str, str],
        *,
        source_file: str,
        line_index: int,
    ) -> Dialog:
        # dialog_id = row.get("key", f"{source_file}_{line_index}")
        language = row.get("language", "en")
        
        # Mapping from Multi-IF CSV instruction IDs to instruction_following.py handler keys
        inst_name_mapping = {
            "change_case:english_capital": "change_case:capital_letter",
            "change_case:english_lowercase": "change_case:lowercase",
            "detectable_content:number_placeholders": "content:placeholder",
            "detectable_content:postscript": "content:postscript",
            "detectable_format:json_format": "format:json_format",
            "detectable_format:multiple_sections": "format:multiple_sections",
            "detectable_format:number_bullet_lists": "format:bullet_list",
            "detectable_format:number_highlighted_sections": "format:number_highlighted_sections",
            "detectable_format:title": "format:title",
            "startend:end_checker": "startend:end_phrase",
        }
        
        dialog_labels = {
            "language": language,
        }
        
        dialog_raw_info = {
            "raw_id": row.get("key", ""),
            "source_file": source_file,
            "line_index": line_index,
        }
        
        turns: List[Turn] = []
        turn_id = 0
        

        # Max 3 turns based on CSV columns observation
        for i in range(1, 4):
            prompt_key = f"turn_{i}_prompt"
            inst_list_key = f"turn_{i}_instruction_id_list"
            kwargs_key = f"turn_{i}_kwargs"
            
            if prompt_key not in row or not row[prompt_key]:
                break
                
            # Parse prompt
            try:
                prompt_data = json.loads(row[prompt_key])
                user_content = prompt_data.get("content", "")
                role = prompt_data.get("role", "user")
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse prompt JSON for dialog {dialog_id} turn {i}")
                continue

            # Parse instructions and kwargs
            metrics = []
            try:
                inst_list_str = row.get(inst_list_key, "[]")
                kwargs_list_str = row.get(kwargs_key, "[]")
                
                inst_list = json.loads(inst_list_str)
                kwargs_list_raw = json.loads(kwargs_list_str)
                
                # kwargs_list_raw is a list of JSON strings
                kwargs_list = [json.loads(k) if isinstance(k, str) else k for k in kwargs_list_raw]
                
                if len(inst_list) != len(kwargs_list):
                    logger.warning(f"Mismatch in instruction/kwargs count for dialog {dialog_id} turn {i}")
                
                for inst_name_raw, inst_args in zip(inst_list, kwargs_list):
                    # Apply mapping if exists, otherwise use the raw name
                    inst_name = inst_name_mapping.get(inst_name_raw, inst_name_raw)
                    
                    metrics.append(
                        MetricConfig(
                            class_name="instruction_following",
                            args={
                                "inst_name": inst_name,
                                "inst_args": inst_args
                            }
                        )
                    )
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse instruction/kwargs JSON for dialog {dialog_id} turn {i}: {e}")

            # User Turn
            turns.append(
                Turn(
                    turn_id=turn_id,
                    role=role,
                    content=user_content,
                )
            )
            turn_id += 1
            
            # Assistant Turn (Evaluated)
            eval_config = TurnEvalConfig(
                do_eval=True if metrics else False,
                metrics=metrics
            )
            
            turns.append(
                Turn(
                    turn_id=turn_id,
                    role="assistant",
                    content=None, # To be generated
                    eval_config=eval_config
                )
            )
            turn_id += 1
            
        dialogue_eval_config = DialogEvalConfig(
            use_reference_history=False,
        )

        return Dialog(
            dialog_id=line_index, # type: ignore
            dialog_labels=dialog_labels,
            dialog_eval_config=dialogue_eval_config,
            dialog_turns=turns,
            dialog_raw_info=dialog_raw_info,
        )

