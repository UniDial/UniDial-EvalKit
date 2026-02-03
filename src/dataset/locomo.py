
import json
import logging
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Any

from .base import BenchmarkContext, BenchmarkDataset
from .schema import Dialog, Turn, TurnEvalConfig, MetricConfig, DialogEvalConfig

logger = logging.getLogger(__name__)

RP_SYSTEM_PROMPT = "Below is a conversation between two people: {} and {}. You are role-playing as {}. The conversation takes place over multiple days and the date of each conversation is written at the beginning of the conversation.\n\n"

QA_PROMPT = """Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {} Short answer:"""

QA_PROMPT_CAT_5 = """Based on the above context, answer the following question.

Question: {} Short answer:"""

class LoCoMoDataset(BenchmarkDataset):
    benchmark_id = "locomo"

    CATEGORY_MAP = {
        1: "Multi-hop",
        2: "Temporal",
        3: "Open-domain",
        4: "Single-hop",
        5: "Adversarial"
    }

    def metric_configs(self) -> List[str]:
        # Default to no metrics or basic metrics.
        # Users can override or use evaluate_qa.py for official evaluation.
        return ["f1_score"]

    def _normalize_raw_data(self, ctx: BenchmarkContext) -> Iterable[Dialog]:
        raw_path = ctx.raw_path
        data_file = raw_path / "locomo10.json"
        
        if not data_file.exists():
            raise FileNotFoundError(f"LoCoMo data file not found at {data_file}")

        with open(data_file, "r", encoding="utf-8") as f:
            samples = json.load(f)

        dialog_counter = -1
        for sample_idx, sample in enumerate(samples):
            sample_id = sample.get("sample_id", "unknown")
            
            # 1. Assign roles: User plays s1, Assistant plays s2
            user_speaker_name = sample["conversation"]["speaker_a"]
            assistant_speaker_name = sample["conversation"]["speaker_b"]

            # 2. System Prompt
            system_prompt = RP_SYSTEM_PROMPT.format(assistant_speaker_name, user_speaker_name, assistant_speaker_name)

            # 3. Build History Turns (Chronological)
            history_turns: List[Turn] = []
            turn_counter = 0
            
            history_turns.append(
                Turn(turn_id=turn_counter, 
                     role="system", 
                     content=system_prompt)
            )
            turn_counter += 1
            
            turn_idx_raw2new: Dict[str, int] = {}
            
            if 'conversation' in sample:
                session_nums = sorted([int(k.split('_')[-1]) for k in sample['conversation'].keys() if 'session' in k and 'date_time' not in k])
                
                # Iterate Chronologically
                for i in session_nums:
                    session_key = f'session_{i}'
                    date_key = f'session_{i}_date_time'
                    
                    date_str = sample['conversation'].get(date_key, "")
                    # Flag to insert date into the first user turn of this session
                    is_first_turn_of_session = True
                    
                    for turn in sample['conversation'][session_key]:
                        speaker = turn['speaker']
                        turn_content = turn['text']
                        if "blip_caption" in turn:
                            turn_content += f' [shared {turn["blip_caption"]}]'
                        
                        # Map speaker to role
                        if speaker == user_speaker_name:
                            current_role = "user"
                        else:
                            current_role = "assistant"
                        
                                                
                        if is_first_turn_of_session:
                            if current_role == "user":
                                turn_content = f"DATE: {date_str}\n\n{turn_content}"
                                is_first_turn_of_session = False
                            else:
                                # Assistant starts the session. Insert a dummy user turn with DATE.
                                history_turns.append(Turn(
                                    turn_id=turn_counter,
                                    role="user",
                                    content=f"DATE: {date_str}",
                                    turn_labels={"raw_date_time": date_str}
                                ))
                                turn_counter += 1
                                is_first_turn_of_session = False
                                # Continue to process the current assistant turn normally
                        
                        
                        history_turns.append(Turn(
                            turn_id=turn_counter,
                            role=current_role,
                            content=turn_content,
                            turn_labels={"raw_id": turn["dia_id"], "raw_date_time": date_str}
                        ))
                        turn_idx_raw2new[turn["dia_id"].strip()] = turn_counter
                        turn_counter += 1
            
            # 4. Generate Dialogs for each QA
            for i, qa in enumerate(sample.get("qa", [])):
                question = qa["question"]
                
                category = qa.get("category", 0)
                category_type = self.CATEGORY_MAP.get(category, "Unknown")
                
                try:
                    answer = str(qa["answer"])
                except: # adversarial eval
                    assert category == 5, f"Adversarial eval should have answer in qa, but got {qa}"
                    answer = ["no information available", "not mentioned"]
                    
                final_question_prompt = self._format_question(question, answer, category)
                
                current_turns = [t.model_copy() for t in history_turns]
                
                # Append Question
                # Logic: Question MUST be asked by User.
                # If last turn is User, append. Else add new User turn.
                if current_turns and current_turns[-1].role == "user":
                    # Append to existing User turn
                    current_turns[-1].content += f"\n\nQuestion: {final_question_prompt}"
                else:
                    # New User turn
                    current_turns.append(Turn(
                        turn_id=turn_counter,
                        role="user",
                        content=f"Question: {final_question_prompt}"
                    ))
                    turn_counter += 1
                
                reference_document = []
                for evidence in qa["evidence"]:
                    if ";" in evidence:
                        evidence = re.split(r"[; ]", evidence)
                        for e in evidence:
                            try:
                                reference_document.append(turn_idx_raw2new[e.strip()])
                            except:
                                print(f"Warning: Evidence {e.strip()} not found in {sample_id}")
                    else:
                        try:
                            reference_document.append(turn_idx_raw2new[evidence.strip()])
                        except:
                            print(f"Warning: Evidence {evidence.strip()} not found in {sample_id}")
                
                
                if category in [2,3,4]:
                    eval_args = {}
                     # Add Assistant turn for answer generation
                    current_turns.append(Turn(
                        turn_id=turn_counter,
                        role="assistant",
                        content=None,
                        reference=answer,
                        reference_document=reference_document,
                        eval_config=TurnEvalConfig(do_eval=True, metrics=[MetricConfig(class_name="f1_score", args=eval_args)]),
                    ))
                elif category in [1]:
                    eval_args = {"multi_answer": True}
                    # Add Assistant turn for answer generation
                    current_turns.append(Turn(
                        turn_id=turn_counter,
                        role="assistant",
                        content=None,
                        reference=answer,
                        reference_document=reference_document,
                        eval_config=TurnEvalConfig(do_eval=True, metrics=[MetricConfig(class_name="f1_score", args=eval_args)]),
                    ))
                else:
                    current_turns.append(Turn(
                        turn_id=turn_counter,
                        role="assistant",
                        content=None,
                        reference=answer,
                        reference_document=reference_document,
                        eval_config=TurnEvalConfig(do_eval=True, metrics=[MetricConfig(class_name="recall", args={"binary": True})]),
                    ))

                dialog_counter += 1
                if dialog_counter > 5:
                    break
                yield Dialog(
                    dialog_id=dialog_counter,
                    dialog_labels={
                        "category": category_type,
                    },
                    dialog_turns=current_turns,
                    dialog_raw_info={
                        "sample_id": sample_id,
                        "source_file": str(data_file),
                        "line_index": sample_idx,
                    },
                    dialog_eval_config=DialogEvalConfig(
                        use_reference_history=True,
                    )
                )
                

    def _format_question(self, question: str, answer: Any, category: int) -> str:
        if category == 2:
            question = question + ' Use DATE of CONVERSATION to answer with an approximate date.'
        elif category == 5:
            fmt = " Select the correct answer: (a) {} (b) {}. "
            if isinstance(answer, list):
                # For adversarial, if it's already a list of keywords, we use them
                question = question + fmt.format(answer[0], answer[1])
            else:
                question = question + fmt.format(answer, 'Not mentioned in the conversation')
            return QA_PROMPT_CAT_5.format(question)
        
        return QA_PROMPT.format(question)
