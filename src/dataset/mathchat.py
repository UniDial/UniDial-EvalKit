import json
from typing import Iterable, List, Dict, Any, Optional
from pathlib import Path
import re

from .base import BenchmarkDataset, BenchmarkContext
from .schema import Dialog, Turn, TurnEvalConfig, MetricConfig, DialogEvalConfig


class MathChatDataset(BenchmarkDataset):
    benchmark_id: str = "mathchat"

    # Evaluation Prompts
    PROMPT_TEMPLATE_ERROR_ANALYSIS = """Evaluate the large language model's ability to identify and correct errors in an attempted solution to a math word problem. The evaluation focuses on the model's comprehension, analytical reasoning, and problem-solving capabilities within the context of mathematical problem-solving. Use the following criteria for scoring:
1. Understanding and Instruction Adherence: Assess how well the AI model understands the given task and follows the instructions. Consider whether the AI model accurately grasps the context and objectives of the task.
2. Identification of the Wrong Attempt: Evaluate the AI model's capability to identify and generate a reasonable and correct analysis of the wrong attempt. Assess the depth and accuracy of the analysis.
3. Correction of the Wrong Solution: Measure the effectiveness of the AI model in correcting the previously wrong solution into a correct one. This not only involves providing the correct answer but also explaining the correct approach to solving the problem, ensuring the explanation is mathematically sound and logically structured.
Scoring Guidelines (1-5 points):
1 point: The model shows very poor understanding and adherence to instructions, provides incorrect or irrelevant analysis of the wrong attempt, and fails to correct the solution or makes it worse.
2 points: The model demonstrates limited understanding and partial adherence to instructions, offers an inaccurate or shallow analysis of the wrong attempt, and corrects the solution with significant errors or misunderstandings.
3 points: The model shows fair understanding and adherence to instructions, provides a moderately accurate analysis of the wrong attempt with some correct elements, and corrects the solution with noticeable errors or logical flaws.
4 points: The model demonstrates good understanding and adherence to instructions, offers a well-reasoned and mostly accurate analysis of the wrong attempt, and corrects the solution effectively with minor mistakes or areas for improvement.
5 points: The model exhibits excellent understanding and strict adherence to instructions, provides a detailed and accurate analysis of the wrong attempt, and corrects the solution perfectly with a clear, logical, and mathematically sound explanation.

Problem: {question}
Correct Solution: {reference}
User's Attempt (Wrong Solution): {wrong_attempt}
Model's Analysis and Correction: {prediction}

For each of the three aspects, provide a score along with a concise rationale for each score. Explain how the AI model's performance aligns with the evaluation criteria and contributes to effectively identifying, analyzing, and correcting the mathematical error.

Finally, output the results in the following JSON format:
{{
    "rationale": "Your detailed scores(1-5) for each criterion and corresponding rationales here...",
    "score": "The sum of the three scores"
}}
"""

    PROMPT_TEMPLATE_P2P = """Evaluate the large language model's ability to generate a problem and solution based on a provided seed problem. The task assesses the model's understanding, creativity in problem generation, and accuracy in solution. Use the following criteria for scoring:
1. Understanding and Instruction Adherence: Assess whether the AI model fully grasps the task and adheres to the instructions given. Consider how well the generated problem aligns with the seed problem's topic or mathematical principles.
2. Problem Relevance and Quality: Evaluate the relevance and quality of the generated problem. Determine if it explores the same topic more deeply or applies the same mathematical principles in a different context, while also assessing the problem's complexity and ingenuity.
3. Solution Accuracy: Check the correctness of the solution provided for the generated problem. Ensure the solution is logically sound, mathematically accurate, and effectively solves the problem.
Scoring Guidelines (1-5):
1 point: The model does not understand the task, generates an unrelated problem, and provides an incorrect or irrelevant solution.
2 points: The model shows limited understanding of the task, creates a problem somewhat related to the seed problem, but the solution has significant errors or is partially irrelevant.
3 points: The model demonstrates a moderate understanding, generates a problem that is relevant and has quality, and provides a solution that is mostly correct with some errors or inconsistencies.
4 points: The model exhibits a good understanding, creates a relevant and well- constructed problem, and provides a solution that is largely correct with minor mistakes.
5 points: The model shows an excellent understanding of the task, generates a highly relevant and challenging problem, and provides a perfectly accurate and comprehensive solution.

Seed Problem: {seed_problem}
Seed Solution: {seed_solution}
Reference (Example): {reference}
Model's Generated Problem and Solution: {prediction}

When scoring, consider the overall effectiveness of the AI model in generating a coherent and related problem-solution pair. Provide a score for each criterion, and a rationale for each score, detailing how the AI model's performance aligns with the evaluation criteria and contributes to the quality of the generated content.

Finally, output the results in the following JSON format:
{{
    "rationale": "Your detailed scores(1-5) for each criterion and corresponding rationales here...",
    "score": "The sum of the three scores"
}}
"""

    def _normalize_raw_data(self, ctx: BenchmarkContext) -> Iterable[Dialog]:
        raw_path = ctx.raw_path
        
        # We expect raw_path to be a directory containing the jsonl files
        if not raw_path.is_dir():
            raise ValueError(f"MathChat raw_path must be a directory: {raw_path}")

        files_map = {
            "follow_up": "follow_up.jsonl",
            "error_correction": "error_correction.jsonl",
            "error_analysis": "error_analysis.jsonl",
            "p2p_generation": "P2P_Generation.jsonl"
        }

        dialog_id = -1
        for task_name, file_name in files_map.items():
            file_path = raw_path / file_name
            if not file_path.exists():
                continue
                
            with open(file_path, "r", encoding="utf-8") as f:
                for line_index, line in enumerate(f):
                    
                    if line_index == 2:
                        break
                    
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    
                    dialog_id += 1         
                    if task_name == "follow_up":
                        yield self._process_follow_up(item, dialog_id, task_name, file_name, line_index,)
                    elif task_name == "error_correction":
                        yield self._process_error_correction(item, dialog_id, task_name, file_name, line_index,)
                    elif task_name == "error_analysis":
                        yield self._process_error_analysis(item, dialog_id, task_name, file_name, line_index,)
                    elif task_name == "p2p_generation":
                        yield self._process_p2p_generation(item, dialog_id, task_name, file_name, line_index,)
                

    def _parse_conversation(self, conversation: str):
        pattern = r'\n*\s*(A:|B:)'
        parts = re.split(pattern, conversation)
        a_dialogue, b_dialogue = [], []
        
        for i in range(1, len(parts), 2):
            speaker = parts[i]
            dialogue = parts[i+1].strip()
            if speaker == 'A:':
                a_dialogue.append(dialogue)
            elif speaker == 'B:':
                b_dialogue.append(dialogue)
        return a_dialogue, b_dialogue

    def _process_follow_up(self, item: Dict[str, Any], dialog_id: str, task_name: str, file_name: str, line_index: int) -> Dialog:
        question = item['question']
        original_answer = item['answer']
        followup_text = item['followup']
        
        a_dialogue, b_dialogue = self._parse_conversation(followup_text)
        
        turns = []
        turn_id = 0
        
        # Turn 0: Original Question
        instruction = "Solve this problem: "
        turns.append(Turn(
            turn_id=turn_id,
            role="user",
            content=instruction + question
        ))
        turn_id += 1
        
        # Turn 1: Original Answer (Eval)
        turns.append(Turn(
            turn_id=turn_id,
            role="assistant",
            reference=original_answer,
            eval_config=TurnEvalConfig(
                do_eval=True,
                metrics=[MetricConfig(class_name="numeric_match")] 
            )
        ))
        turn_id += 1
        
        # Follow-up turns
        for i in range(len(a_dialogue)):
            # User Turn
            turns.append(Turn(
                turn_id=turn_id,
                role="user",
                content=a_dialogue[i]
            ))
            turn_id += 1
            
            # Assistant Turn
            ref = b_dialogue[i] if i < len(b_dialogue) else None
            turns.append(Turn(
                turn_id=turn_id,
                role="assistant",
                reference=ref,
                eval_config=TurnEvalConfig(
                    do_eval=True,
                    metrics=[MetricConfig(class_name="numeric_match")]
                )
            ))
            turn_id += 1
        
        dialog_labels: Dict[str, object] = {
            "task_type": task_name,
        }
        
        dialog_raw_info: Dict[str, object] = {
            "source_file": file_name,
            "line_index": line_index,
        }
        
        dialogue_eval_config = DialogEvalConfig(
            use_reference_history=True, # 决定了历史assistant语句是选择reference还是选择content
        )

        dialog = Dialog(
            dialog_id=dialog_id, 
            dialog_labels=dialog_labels, 
            dialog_eval_config=dialogue_eval_config,
            dialog_turns=turns,
            dialog_raw_info=dialog_raw_info
        )

            
        return dialog

    def _process_error_correction(self, item: Dict[str, Any], dialog_id: str, task_name: str, file_name: str, line_index: int) -> Dialog:
        a_dialogue, b_dialogue = self._parse_conversation(item['error_correction'])
        
        turns = []
        turn_id = 0
        
        turns.append(Turn(
            turn_id=turn_id,
            role="user",
            content="Please give me a math problem and I will answer that. You need to analyze my solution and correct it if I make errors."
        ))
        turn_id += 1
        
        # Context Turn (Pre-filled Assistant)
        turns.append(Turn(
            turn_id=turn_id,
            role="assistant",
            content=a_dialogue[0] if a_dialogue else ""
        ))
        turn_id += 1
        
        # Context Turn (User - the one with errors?)
        turns.append(Turn(
            turn_id=turn_id,
            role="user",
            content=b_dialogue[0] if b_dialogue else ""
        ))
        turn_id += 1
        
        # Final Turn to Generate
        turns.append(Turn(
            turn_id=turn_id,
            role="assistant",
            reference=item['answer'],
            eval_config=TurnEvalConfig(
                do_eval=True,
                metrics=[MetricConfig(class_name="numeric_match")]
            )
        ))
        
        dialog_labels: Dict[str, object] = {
            "task_type": task_name,
        }
        
        dialog_raw_info: Dict[str, object] = {
            "source_file": file_name,
            "line_index": line_index,
        }
        
        dialogue_eval_config = DialogEvalConfig(
            use_reference_history=True, # 决定了历史assistant语句是选择reference还是选择content
        )

        dialog = Dialog(
            dialog_id=dialog_id, 
            dialog_labels=dialog_labels, 
            dialog_eval_config=dialogue_eval_config,
            dialog_turns=turns,
            dialog_raw_info=dialog_raw_info
        )

        return dialog

    def _process_error_analysis(self, item: Dict[str, Any], dialog_id: str, task_name: str, file_name: str, line_index: int) -> Dialog: 
        a_dialogue, b_dialogue = self._parse_conversation(item.get('error_correction', '')) 
        
        turns = []
        turn_id = 0
        
        turns.append(Turn(
            turn_id=turn_id,
            role="user",
            content="Please give me a math problem and I will answer that. You need to analyze my solution and correct it if I make errors."
        ))
        turn_id += 1
        
        turns.append(Turn(
            turn_id=turn_id,
            role="assistant",
            content=a_dialogue[0] if a_dialogue else ""
        ))
        turn_id += 1
        
        turns.append(Turn(
            turn_id=turn_id,
            role="user",
            content=b_dialogue[0] if b_dialogue else ""
        ))
        turn_id += 1
        
        # Capture context for rendering in prompt
        # We need to pass the context: Question (a[0]), Wrong Attempt (b[0]), Correct Solution (item['answer'])
        # Dynamic config source can be used to pass extra data to metrics
        
        dynamic_config = {
            "task_name": task_name, 
            "question": item["question"],
            "wrong_attempt": b_dialogue[0] if b_dialogue else "",
        }
        
        # Final Turn
        turns.append(Turn(
            turn_id=turn_id,
            role="assistant",
            reference=item['answer'], 
            eval_config=TurnEvalConfig(
                do_eval=True,
                dynamic_config_source=dynamic_config,
            )
        ))
        
        dialog_labels: Dict[str, object] = {
            "task_type": task_name,
        }
        
        dialog_raw_info: Dict[str, object] = {
            "source_file": file_name,
            "line_index": line_index,
        }
        
        dialogue_eval_config = DialogEvalConfig(
            use_reference_history=True, # 决定了历史assistant语句是选择reference还是选择content
        )

        dialog = Dialog(
            dialog_id=dialog_id, 
            dialog_labels=dialog_labels, 
            dialog_eval_config=dialogue_eval_config,
            dialog_turns=turns,
            dialog_raw_info=dialog_raw_info
        )

        
        return dialog

    def _process_p2p_generation(self, item: Dict[str, Any], dialog_id: str, task_name: str, file_name: str, line_index: int) -> Dialog:
        instruction = 'Your task is to create a new math problem based on a given seed problem. The generated problems should either explore the same topic in greater depth or apply the same mathematical principles in a different context. Each problem should be accompanied by a detailed solution that demonstrates the correct application of the mathematical principles involved.'
        
        turns = []
        turn_id = 0
        
        turns.append(Turn(
            turn_id=turn_id,
            role="user",
            content=instruction
        ))
        turn_id += 1
        
        turns.append(Turn(
            turn_id=turn_id,
            role="assistant",
            content='Understood, please give me the seed problem.'
        ))
        turn_id += 1
        
        seed_content = 'Seed problem: ' + item['question'] + ' Solution: ' + item['answer'] + '\n'
        turns.append(Turn(
            turn_id=turn_id,
            role="user",
            content=seed_content
        ))
        turn_id += 1
        
        dynamic_config = {
            "task_name": task_name,
            "seed_problem": item['question'],
            "seed_solution": item['answer'],
        }
        
        # Final Turn
        turns.append(Turn(
            turn_id=turn_id,
            role="assistant",
            reference=item.get('new_problem'),
            eval_config=TurnEvalConfig(
                do_eval=True,
                dynamic_config_source=dynamic_config
            )
        ))
        
        dialog_labels: Dict[str, object] = {
            "task_type": task_name,
        }
        
        dialog_raw_info: Dict[str, object] = {
            "source_file": file_name,
            "line_index": line_index,
        }
        
        dialogue_eval_config = DialogEvalConfig(
            use_reference_history=True, # 决定了历史assistant语句是选择reference还是选择content
        )

        dialog = Dialog(
            dialog_id=dialog_id, 
            dialog_labels=dialog_labels, 
            dialog_eval_config=dialogue_eval_config,
            dialog_turns=turns,
            dialog_raw_info=dialog_raw_info
        )
        
        return dialog
      
    def get_eval_config_for_turn(self, turn: Turn) -> List[MetricConfig]:
        
        if turn.eval_config.dynamic_config_source.get("task_name", "") == "error_analysis":
            return [MetricConfig(class_name="llm_judge", args={"template_name": "math_chat_error_analysis", "question": turn.eval_config.dynamic_config_source.get("question", ""), "wrong_attempt": turn.eval_config.dynamic_config_source.get("wrong_attempt", "")})]
        elif turn.eval_config.dynamic_config_source.get("task_name", "") == "p2p_generation":
            return [MetricConfig(class_name="llm_judge", args={"template_name": "math_chat_p2p_generation", "seed_problem": turn.eval_config.dynamic_config_source.get("seed_problem", ""), "seed_solution": turn.eval_config.dynamic_config_source.get("seed_solution", "")})]
        else:
            return turn.eval_config.metrics
        
    def prompt_template_render(self, template_name: str, **kwargs) -> str:
        if template_name == "math_chat_error_analysis":

            return self.PROMPT_TEMPLATE_ERROR_ANALYSIS.format(
                question=kwargs.get("question", ""),
                wrong_attempt=kwargs.get("wrong_attempt", ""),
                reference=kwargs.get("reference", ""),
                prediction=kwargs.get("prediction", "")
            )
            
        elif template_name == "math_chat_p2p_generation":
            
            return self.PROMPT_TEMPLATE_P2P.format(
                seed_problem=kwargs.get("seed_problem", ""),
                seed_solution=kwargs.get("seed_solution", ""),
                reference=kwargs.get("reference", ""),
                prediction=kwargs.get("prediction", "")
            )
            
        raise ValueError(f"Unknown template name: {template_name}")


    def metric_configs(self) -> List[str]:
        return {"llm_judge": {"min_score": 1, "max_score": 15}, "numeric_match":{}}
    