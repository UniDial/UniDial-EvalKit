import json
from pathlib import Path
from typing import Iterable, Any, Dict, List, Optional
from collections import defaultdict

from .base import BenchmarkDataset, BenchmarkContext
from .schema import Dialog, Turn, TurnEvalConfig, MetricConfig, DialogEvalConfig


class LongMemEvalDataset(BenchmarkDataset):
    benchmark_id: str = "longmemeval"

    def prompt_template_render(
        self,
        prediction: str,
        reference: str,
        question_type: str,
        raw_question: str,
        abstention: bool = False,
    ) -> str:
        """
        Render the prompt for LLMJudge based on question type and abstention status.
        Adapted from evaluate_qa.py get_anscheck_prompt.
        """
        
        # Determine which template to use
        if abstention:
            template = (
                "I will give you an unanswerable question, an explanation, and a response from a model. "
                "Please answer yes if the model correctly identifies the question as unanswerable. "
                "The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\n"
                "Question: {raw_question}\n\n"
                "Explanation: {reference}\n\n"
                "Model Response: {prediction}\n\n"
                "Does the model correctly identify the question as unanswerable? "
                "Answer with JSON format: {{\"Score\": 1, \"Rationale\": \"reasoning...\"}} if yes, or {{\"Score\": 0, \"Rationale\": \"reasoning...\"}} if no."
            )
        else:
            if question_type in ['single-session-user', 'single-session-assistant', 'multi-session']:
                template = (
                    "I will give you a question, a correct answer, and a response from a model. "
                    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                    "If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. "
                    "If the response only contains a subset of the information required by the answer, answer no. \n\n"
                    "Question: {raw_question}\n\n"
                    "Correct Answer: {reference}\n\n"
                    "Model Response: {prediction}\n\n"
                    "Is the model response correct? "
                    "Answer with JSON format: {{\"Score\": 1, \"Rationale\": \"reasoning...\"}} if yes, or {{\"Score\": 0, \"Rationale\": \"reasoning...\"}} if no."
                )
            elif question_type == 'temporal-reasoning':
                template = (
                    "I will give you a question, a correct answer, and a response from a model. "
                    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                    "If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. "
                    "If the response only contains a subset of the information required by the answer, answer no. "
                    "In addition, do not penalize off-by-one errors for the number of days. "
                    "If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\n"
                    "Question: {raw_question}\n\n"
                    "Correct Answer: {reference}\n\n"
                    "Model Response: {prediction}\n\n"
                    "Is the model response correct? "
                    "Answer with JSON format: {{\"Score\": 1, \"Rationale\": \"reasoning...\"}} if yes, or {{\"Score\": 0, \"Rationale\": \"reasoning...\"}} if no."
                )
            elif question_type == 'knowledge-update':
                template = (
                    "I will give you a question, a correct answer, and a response from a model. "
                    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                    "If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\n"
                    "Question: {raw_question}\n\n"
                    "Correct Answer: {reference}\n\n"
                    "Model Response: {prediction}\n\n"
                    "Is the model response correct? "
                    "Answer with JSON format: {{\"Score\": 1, \"Rationale\": \"reasoning...\"}} if yes, or {{\"Score\": 0, \"Rationale\": \"reasoning...\"}} if no."
                )
            elif question_type == 'single-session-preference':
                template = (
                    "I will give you a question, a rubric for desired personalized response, and a response from a model. "
                    "Please answer yes if the response satisfies the desired response. Otherwise, answer no. "
                    "The model does not need to reflect all the points in the rubric. "
                    "The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\n"
                    "Question: {question}\n\n"
                    "Rubric: {reference}\n\n"
                    "Model Response: {prediction}\n\n"
                    "Is the model response correct? "
                    "Answer with JSON format: {{\"Score\": 1, \"Rationale\": \"reasoning...\"}} if yes, or {{\"Score\": 0, \"Rationale\": \"reasoning...\"}} if no."
                )
            else:
                # Fallback to standard
                template = (
                    "I will give you a question, a correct answer, and a response from a model. "
                    "Please answer yes if the response contains the correct answer. Otherwise, answer no. \n\n"
                    "Question: {question}\n\n"
                    "Correct Answer: {reference}\n\n"
                    "Model Response: {prediction}\n\n"
                    "Is the model response correct? "
                    "Answer with JSON format: {{\"Score\": 1, \"Rationale\": \"reasoning...\"}} if yes, or {{\"Score\": 0, \"Rationale\": \"reasoning...\"}} if no."
                )

        return template.format(raw_question=raw_question, reference=reference, prediction=prediction)

    def metric_configs(self) -> List[str]:
        return {"llm_judge": {"min_score": 0, "max_score": 1}}
    

    def _normalize_raw_data(self, ctx: BenchmarkContext) -> Iterable[Dialog]:
        raw_path = ctx.raw_path
        if raw_path.is_dir():
             target_file = raw_path / "longmemeval_s_cleaned.json" # _m_ file is too large, skip for now
             if target_file.exists():
                 data_file = target_file
             else:
                 data_file = raw_path
        else:
            data_file = raw_path

        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if "_s" in data_file.name:
            context_type = "longmemeval_s"
        elif "_m" in data_file.name:
            context_type = "longmemeval_m"
        elif "_oracle":
            context_type = "longmemeval_oracle"
        
        for i,item in enumerate(data):
            # if i==5:
            #     break
            
            yield self._build_dialog(
                item,
                dialog_id=i,
                source_file=data_file.name,
                context_type=context_type,
                )
    
    def _build_dialog(self, item: Dict[str, Any],
                      dialog_id: int,
                      source_file: str,
                      context_type: str) -> Dialog:
        question_id = item["question_id"]
        haystack_sessions = item["haystack_sessions"]
        haystack_dates = item["haystack_dates"]
        haystack_session_ids = item["haystack_session_ids"]
        
        question = item["question"]
        question_date = item.get("question_date", "")
        question_type = item["question_type"]
        answer = str(item["answer"])
        
        dialog_labels: Dict[str, object] = {
            "task_type": context_type,
            "task_subtype":  question_type   
        }
        
        dialog_raw_info: Dict[str, object] = {
            "raw_id": question_id,
            "source_file": source_file,
        }

                
        dialogue_eval_config = DialogEvalConfig(
            use_reference_history=True, # Determines whether history assistant messages use reference or content
        )
        
        dialog_turns: List[Turn] = []
        turn_counter = 0
        
        raw_session_idx2new_turn_idx: Dict[str, List[int]] = defaultdict(list)

        # Process history sessions
        for i, session in enumerate(haystack_sessions):
            session_date = haystack_dates[i]
            session_id = haystack_session_ids[i]
            
            for j, turn_data in enumerate(session):
                
                content = turn_data["content"]
                # Session time reflects in the first sentence of each session
                if j == 0:
                    content = f"[{session_date}] {content}"
                
                turn = Turn(
                    turn_id=turn_counter,
                    role=turn_data["role"],
                    content=content,
                    turn_labels={
                        "raw_session_id": session_id,
                        "raw_session_date": session_date,
                        "raw_session_index": i,
                        }
                )
                raw_session_idx2new_turn_idx[session_id].append(turn_counter)
                dialog_turns.append(turn)
                turn_counter += 1

        # Process final question
        final_content = f"[{question_date}] {question}" if question_date else question
        
        is_abstention = "_abs" in str(question_id)
        
        metric_args = {
            "question_type": question_type,
            "abstention": is_abstention,
            "raw_question": question
        }
        
        eval_config = TurnEvalConfig(
            do_eval=True,
            metrics=[
                MetricConfig(
                    class_name="llm_judge",
                    args=metric_args
                )
            ]
        )
        
        reference_document = []
        for sid in item["answer_session_ids"]:
            reference_document.extend(raw_session_idx2new_turn_idx[sid])

        final_user_turn = Turn(
            turn_id=turn_counter,
            role="user",
            content=final_content,
            turn_labels={"raw_question_date": question_date}
        )
        dialog_turns.append(final_user_turn)
        turn_counter += 1
        
        final_assistant_turn = Turn(
            turn_id=turn_counter,
            role="assistant",
            content=None,
            reference=answer,
            reference_document=reference_document,
            eval_config=eval_config,
        )
        dialog_turns.append(final_assistant_turn)
        
        dialog = Dialog(
            dialog_id=dialog_id,
            dialog_labels=dialog_labels, 
            dialog_eval_config=dialogue_eval_config,
            dialog_turns=dialog_turns,
            dialog_raw_info=dialog_raw_info
        )
        
        return dialog

