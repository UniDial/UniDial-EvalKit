import json
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional
import re

from .base import BenchmarkDataset, BenchmarkContext
from .schema import Dialog, Turn, TurnEvalConfig, MetricConfig, DialogEvalConfig


class MemoryCodeDataset(BenchmarkDataset):
    benchmark_id: str = "memorycode"

    PREAMBLE = """You are {mentee}, a new software engineer at {company}. Your mentor {mentor} has given you specific coding guidelines that you must follow.

    Do not acknowledge. Only generate Python code and nothing else before or after. Do not explain the code. Do not ask for more information but directly give the answer. 
    """
    
    def _parse_conversation(self, conversation_text: str, mentor_name: str, mentee_name: str, session_idx: int):
        """Parse conversation text into list of (speaker, content, session_idx) tuples."""
        
        # Simple parsing strategy:
        # Regex to find "Name:" followed by content until next "Name:" or end.
        
        pattern = f"({mentor_name}|{mentee_name}):"
        parts = re.split(pattern, conversation_text)
        # parts[0] is empty or text before first speaker
        # parts[1] is speaker1, parts[2] is content1
        # parts[3] is speaker2, parts[4] is content2

        dialogue_turns = []
        if len(parts) > 1:
            for i in range(1, len(parts), 2):
                speaker = parts[i]
                content = parts[i+1].strip()
                dialogue_turns.append((speaker, content, session_idx))
        else:
            # Fallback if parsing fails, treat whole text as user content? 
            # Or just return empty list and let caller handle.
            # But the task requires expanding history.
            pass
            
        return dialogue_turns

    def _normalize_raw_data(self, ctx: BenchmarkContext) -> Iterable[Dialog]:
        raw_path = ctx.raw_path
        
        # Iterate over all dialogue_*.json files
        # Using glob to find files matching the pattern
        dialog_files = sorted(list(raw_path.glob("dialogue_*.json")), key=lambda p: int(p.stem.split("_")[1]))
        
        dialog_id = -1
        for file_path in dialog_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

            raw_dialog_id = int(file_path.stem.split("_")[1])
            context = data.get("context", {})
            sessions = data.get("sessions", [])

            mentee = context.get("mentee", "Mentee")
            mentor = context.get("mentor", "Mentor")
            company = context.get("company", "Company")
            mentor_persona = context.get("mentor_persona", "Mentor Persona")
            mentee_persona = context.get("mentee_persona", "Mentee Persona")
            
            preamble = self.PREAMBLE.format(mentee=mentee, mentor=mentor, company=company)
            
            # Accumulate all turns from all sessions for history
            history_turns = []
            
            total_sessions = len(sessions)
            history_tag = "short" if total_sessions <= 15 else "long"

            for session_idx, session in enumerate(sessions):
                current_session_text = session.get("text", "")
                
                # Parse current session text into turns
                session_parsed_turns = self._parse_conversation(current_session_text, mentor, mentee, session_idx)
                
                # Add to history
                history_turns.extend(session_parsed_turns)
                
                # Only evaluate on the last session for history evaluation
                if session_idx == total_sessions - 1:
                    history_eval_queries = session.get("history_eval_query", [])
                    history_regexes = session.get("history_regex", [])
                    
                    if history_eval_queries and history_regexes and len(history_eval_queries) == len(history_regexes):
                        for q_idx, (query, regex) in enumerate(zip(history_eval_queries, history_regexes)):
                            
                            dialog_id += 1
                            if dialog_id == 5:
                                break
                            yield self._create_dialog(
                                dialog_id=dialog_id,
                                raw_dialog_id=raw_dialog_id,
                                query_idx=q_idx,
                                task_type="history_eval",
                                preamble=preamble,
                                history_turns=history_turns, 
                                mentor=mentor,
                                mentee=mentee,
                                query=query,
                                regex=regex,
                                file_name=file_path.name,
                                history_tag=history_tag,
                                company=company,
                                mentor_persona=mentor_persona,
                                mentee_persona=mentee_persona,
                            )
            if dialog_id == 5:
                break
    def _create_dialog(
        self, 
        dialog_id: int,
        raw_dialog_id: int, #
        query_idx: int, #
        task_type: str,#
        preamble: str, 
        history_turns: List[tuple], 
        mentor: str, 
        mentee: str,
        query: str, 
        regex: Any,
        file_name: str, #
        history_tag: str, #
        mentor_persona: str,
        mentee_persona: str,
        company: str,
    ) -> Dialog:
        
        
        dialog_labels = {
            "task_type": task_type,
            "task_subtype": history_tag
        }
        
        dialog_raw_info = {
            "source_file": file_name,
            "original_dialog_id": raw_dialog_id,
            "query_idx": query_idx,
            "mentor": mentor,
            "mentee": mentee,
            "company": company,
            "mentor_persona": mentor_persona,
            "mentee_persona": mentee_persona,
        }
        
        dialog_eval_config = DialogEvalConfig(
            use_reference_history=True 
        )
        
        turns = []
        turn_id = 0
        
        # Turn 0: System Prompt (Preamble)        
        turns.append(Turn(
            turn_id=turn_id,
            role="system",
            content=preamble
        ))
        turn_id += 1
        
        # History Turns
        for speaker, content, turn_session_idx in history_turns:
            # Map speaker name to role
            role = "user" if speaker == mentor else "assistant"
            
            turns.append(Turn(
                turn_id=turn_id,
                role=role,
                content=content,
                turn_labels={"raw_session_id": turn_session_idx}
            ))
            turn_id += 1
            
        # Final User Turn: The Evaluation Query
        final_query_template = "Based on information provided, write a {eval_query}. Do not provide example usage. You must follow all the latest coding guidelines provided by your mentor, including any possible updates."
        
        final_content = final_query_template.format(eval_query=query)
        
        turns.append(Turn(
            turn_id=turn_id,
            role="user",
            content=final_content
        ))
        turn_id += 1
        
        # Assistant Turn (to be generated and evaluated)
        # Pass the regex config to the metric
        turns.append(Turn(
            turn_id=turn_id,
            role="assistant",
            eval_config=TurnEvalConfig(
                do_eval=True,
                dynamic_config_source={
                    "regex": regex,
                },
            )
        ))
        

        return Dialog(
            dialog_id=dialog_id,
            dialog_labels=dialog_labels,
            dialog_eval_config=dialog_eval_config,
            dialog_turns=turns,
            dialog_raw_info=dialog_raw_info
        )

    def metric_configs(self) -> Dict[str, Any]:
        return {"code_match": {}}

    def get_eval_config_for_turn(self, turn: Turn) -> List[MetricConfig]:
        """
        Get evaluation configuration for a specific turn.
        Subclasses can override this to provide dynamic evaluation configuration.
        """
        return [MetricConfig(class_name="code_match", args={"regex": turn.eval_config.dynamic_config_source.get("regex", "")})]