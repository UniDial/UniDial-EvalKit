import json
import csv
from pathlib import Path
from typing import Iterable, Any, Dict, List, Optional

from .base import BenchmarkDataset, BenchmarkContext
from .schema import Dialog, Turn, TurnEvalConfig, MetricConfig, DialogEvalConfig

class PersonaMemDataset(BenchmarkDataset):
    benchmark_id: str = "personamem"

    def _normalize_raw_data(self, ctx: BenchmarkContext) -> Iterable[Dialog]:
        raw_path = ctx.raw_path
        
        # Define available splits
        splits = ["32k", "128k", "1M"]
        
        dialog_id = 0
        # If raw_path is a directory, iterate through all splits
        if raw_path.is_dir():
            for split in splits:
                if split=="128k":
                    break
                questions_path = raw_path / f"questions_{split}.csv"
                context_path = raw_path / f"shared_contexts_{split}.jsonl"
                
                if questions_path.exists() and context_path.exists():
                    
                    contexts = {}
                    if context_path.exists():
                        with open(context_path, "r", encoding="utf-8") as f:
                            for line in f:
                                if not line.strip(): continue
                                item = json.loads(line)
                                # print(list(item.keys())[0])
                                # exit(0)
                                cid = list(item.keys())[0]
                                contexts[cid] = item[cid]

                    # Load questions
                    with open(questions_path, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for i, row in enumerate(reader):
                            # if i==5:
                            #     break
                            # Parse row
                            question_type = row["question_type"]
                            question = row["user_question_or_message"]
                            answer = row["correct_answer"]
                            options = row["all_options"]
                            
                            shared_context_id = row["shared_context_id"]
                            end_index = int(row["end_index_in_shared_context"])
                            
                            # Construct History
                            dialog_turns: List[Turn] = []
                            turn_counter = 0
                            
                            full_context_text = contexts.get(shared_context_id, [])
                            history_text = full_context_text[:end_index]
                            
                            
                            # Let's add context as the first User turn
                            if history_text:
                                for turn in history_text:    
                            
                                    dialog_turns.append(Turn(
                                        turn_id=turn_counter,
                                        role=turn["role"],
                                        content=turn["content"],
                                    ))
                                    turn_counter += 1
                                
                            
                            # Turn 1: Question
                            # Construct input with instructions
                            instructions = "Find the most appropriate model response and give your final answer (a), (b), (c), or (d) after the special token <final_answer>."
                            final_input = f"{question}\n\n{instructions}\n\n{options}"
                            
                            user_turn = Turn(
                                turn_id=turn_counter,
                                role="user",
                                content=final_input,
                            )
                            dialog_turns.append(user_turn)
                            turn_counter += 1
                            
                            # Turn 2: Assistant (Placeholder for Eval)
                            eval_config = TurnEvalConfig(
                                do_eval=True,
                            )
                            
                            assistant_turn = Turn(
                                turn_id=turn_counter,
                                role="assistant",
                                content=None,
                                reference=answer,
                                eval_config=eval_config
                            )
                            dialog_turns.append(assistant_turn)
                            
                            
                            dialog_labels: Dict[str, object] = {
                                "task_type": split,
                                "task_subtype": question_type,

                            }
                            
                            dialog_raw_info: Dict[str, object] = {
                                "question_file": questions_path.name,
                                "context_file": context_path.name,
                                "persona_id":row["persona_id"],
                                "question_id": row["question_id"],
                                "shared_context_id": shared_context_id,
                                "topic": row["topic"],
                                "distance_to_ref_proportion_in_context": row["distance_to_ref_proportion_in_context"]
                            }
                            
                            dialogue_eval_config = DialogEvalConfig(
                                use_reference_history=True, # 决定了历史assistant语句是选择reference还是选择content
                            )

                            dialog = Dialog(
                                dialog_id=dialog_id, 
                                dialog_labels=dialog_labels, 
                                dialog_eval_config=dialogue_eval_config,
                                dialog_turns=dialog_turns,
                                dialog_raw_info=dialog_raw_info
                            )
                            dialog_id += 1

                            yield dialog
                                
        else:
            raise ValueError(f"Unexpected raw_path format: {raw_path}")
        

    def get_eval_config_for_turn(self, turn: Turn) -> List[MetricConfig]:
        if not turn.eval_config.do_eval:
            return []
        
        return [
            MetricConfig(
                class_name="exact_match", # Hypothertical metric for parsing <final_answer>
                args={
                    "split_special_start_token": "<final_answer>",
                    "split_special_end_token": "</final_answer>"
                }
            )
        ]
        
    def metric_configs(self) -> List[str]:
        return ["exact_match"]
    

