from __future__ import annotations

import json
import warnings
import logging
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Any
from nltk import sent_tokenize

from .base import BenchmarkContext, BenchmarkDataset
from .data_utils import iter_jsonl_lines
from .schema import Dialog, Turn, TurnEvalConfig, MetricConfig, DialogEvalConfig

logger = logging.getLogger(__name__)




def _iter_mteval_files(raw_path: Path) -> Iterator[Path]:
    
    for p in sorted(raw_path.glob("*.jsonl")):
        if not p.name.startswith(".") and not p.name.startswith("_"):
            yield p


# def _as_text(value: Optional[object]) -> str:
#     if value is None:
#         return ""
#     return value if isinstance(value, str) else str(value)


class MTEvalDataset(BenchmarkDataset):
    benchmark_id = "mt_eval"

    def prompt_template_render(self, 
                               template_name: str, 
                               history_messages: List[Dict[str, Any]], 
                               prediction: str, 
                               reference: str, 
                               constraints: Optional[List[str]] = None) -> str: 
        
        prompt_templates = self.prompt_templates()
        template = prompt_templates.get(template_name)
        if not template:
            raise ValueError(f"Template {template_name} not found")
        
        word_count = len(prediction.split())
        sent_count = len(sent_tokenize(prediction))
        
        # Handle constraints: convert to string format or use empty string if not provided
        if constraints:
            if len(constraints) > 1:
                constraints_str = "\n".join([f"{i}. {c}" for i, c in enumerate(constraints, 1)])
            else:
                constraints_str = constraints[0] if constraints else ""
        else:
            constraints_str = ""
                
        if template_name != "mt_bench_evaluation":
            return template.format(content=reference, response=prediction, num_words=word_count, num_sent=sent_count, constraints=constraints_str)
        else:
            conversation = []
            
            for x in history_messages[-3:]:
                if x["role"] == "user":
                    conversation.append(f"User: {x['content']}")
                else:
                    conversation.append(f"Assistant: {x['content']}")
            conversation.append(f"Assistant: {prediction}")
            conversation = "\n".join(conversation)
            return template.format(conversation=conversation, num_words=word_count, num_sent=sent_count)
        
        
    def prompt_templates(self) -> Dict[str, str]:
        
        return {
            "refinement_single_evaluation": (
                "Evaluate the response provided below to determine if it meets the specified constraints related to the following article. "
                "Provide an integer score from 1 to 10, taking into account its helpfulness, relevance, accuracy, depth, creativity, and how well it conforms to the constraints. "
                "For constraints related to word and sentence counts, you must use my provided counts to judge whether the response fulfills the constraint. "
                "Before giving your score, you should first provide a rationale to explain it. \n\n"
                "Article to Evaluate Against:\n"
                "{content}\n\n"
                "Response to Evaluate:\n"
                "{response}\n\n"
                "Number of words in response: {num_words} \n"
                "Number of sentences in response: {num_sent} \n\n"
                "Constraints:\n"
                "{constraints}\n\n"
                "The evaluation must be structured in the following JSON format:\n"
                "\n"
                "```json\n"
                "{{\n"
                '  "Rationale": "<Explain the rationale of your score.>",\n'
                '  "Score": <An integer score from 1 to 10.>\n'
                "}}\n"
                "```"
            ),
            "refinement_multi_evaluation": (
                "Evaluate the response provided below to determine if it meets the specified constraints related to the following article. "
                "Provide an integer score from 1 to 10, taking into account its helpfulness, relevance, accuracy, depth, creativity, and how well it conforms to the constraints. "
                "You should ignore any earlier constraints that contradict to the latter constraints. "
                "For constraints related to word and sentence counts, you must use my provided counts to judge whether the response fulfills the constraint. "
                "Before giving your score, you should first provide a rationale to explain it. \n\n"
                "Article to Evaluate Against:\n"
                "{content}\n\n"
                "Response to Evaluate:\n"
                "{response}\n\n"
                "Number of words in response: {num_words} \n"
                "Number of sentences in response: {num_sent} \n\n"
                "Constraints:\n"
                "{constraints}\n\n"
                "The evaluation must be structured in the following JSON format:\n"
                "\n"
                "```json\n"
                "{{\n"
                '  "Rationale": "<Explain the rationale of your score.>",\n'
                '  "Score": <An integer score from 1 to 10.>\n'
                "}}\n"
                "```"
            ),
            "mt_bench_evaluation": (
                "Evaluate the last response of the assistant in the conversation provided below to determine if it meets the specified constraints related to the following article. "
                "Provide an integer score from 1 to 10, taking into account its helpfulness, relevance, accuracy, depth, creativity, and level of detail of the assistant's response. "
                "For user queries that are related to word and sentence counts, you must use my provided counts to judge whether the response fulfills the requirement. "
                "Before giving your score, you should first provide a rationale to explain it. \n\n"
                "Conversation:\n"
                "{conversation}\n\n"
                "Number of words in response: {num_words} \n"
                "Number of sentences in response: {num_sent} \n\n"
                "The evaluation must be structured in the following JSON format:\n"
                "\n"
                "```json\n"
                "{{\n"
                '  "Rationale": "<Explain the rationale of your score.>",\n'
                '  "Score": <An integer score from 1 to 10.>\n'
                "}}\n"
                "```"
            ),
            "expansion_evaluation": (
                "Evaluate the response provided below to determine if it meets the specified constraints related to the following article. "
                "Provide an integer score from 1 to 10, taking into account its helpfulness, relevance and accuracy. "
                "Before giving your score, you should first provide a rationale to explain it. \n\n"
                "Article to Evaluate Against:\n"
                "{content}\n\n"
                "Constraints:\n"
                "{constraints}\n\n"
                "Response to Evaluate:\n"
                "{response}\n\n"
                "The evaluation must be structured in the following JSON format:\n"
                "\n"
                "```json\n"
                "{{\n"
                '  "Rationale": "<Explain the rationale of your score.>",\n'
                '  "Score": <An integer score from 1 to 10.>\n'
                "}}\n"
                "```"
            ),
        }

    def metric_configs(self) -> List[str]:
        return ["llm_judge", "instruction_following", "precision"]
    
    
    def _normalize_raw_data(self, ctx: BenchmarkContext) -> Iterable[Dialog]:
        raw_path = ctx.raw_path
        logger.info(f"MTEvalDataset: checking raw_path: {raw_path.absolute()}")
        if not raw_path.exists():
            logger.error(f"MTEvalDataset: raw_path does not exist: {raw_path}")
            raise FileNotFoundError(raw_path)

        dialog_index = -1
        for file_path in _iter_mteval_files(raw_path):
            
            if file_path.name.endswith("documents.jsonl"):
                with file_path.open("r", encoding="utf-8") as f:
                    tmp_documents = [json.loads(row) for row in f]
                self.documents = {}
                for i, doc in enumerate(tmp_documents):
                    self.documents[i] = doc["gen_resp"].split("\n\n", 1)[1]
                continue

            task_name = file_path.stem
            for line_index, line in enumerate(iter_jsonl_lines(file_path)):
                # for debugging
                # if line_index == 2:
                #     break 
                
                data = json.loads(line)
                dialog_index += 1
                yield self._build_dialog(
                    data,
                    dialog_id=dialog_index,
                    task_name=task_name,
                    source_file=file_path.name,
                    line_index=line_index,
                )

    def _build_dialog(
        self,
        data: Dict,
        *,
        dialog_id: int,
        task_name: str,
        source_file: str,
        line_index: int,
    ) -> Dialog:
        conv = data.get("conv") or []

        dialog_labels: Dict[str, object] = {
            "task_type": task_name.split("_")[0],
            "task_subtype": task_name.split("_")[1],
    
        }
        
        dialog_raw_info: Dict[str, object] = {
            "raw_id": data.get("id"),
            "source_file": source_file,
            "line_index": line_index,
        }

        # turn_ids: List[Optional[str]] = []
        # instructions: List[Optional[str]] = []

        turns: List[Turn] = []
        turn_id = 0

        prev_task_type = ""
        constraints = []
        
        for item in conv:
            user_text = item.get("user")
            turns.append(Turn(turn_id=turn_id, role="user", content=user_text))
            turn_id += 1

            assistant_text = item.get("sys")
            do_eval = bool(item.get("do_inference", False))
            
            # prepare for constraints
            raw_turn_id = item.get("id")

            # if task_name in ["refinement_multi", "refinement_single", "expansion_multi", "expansion_single"]:

            
            reference_document = None
            cur_task_type = None
            if task_name in ["refinement_multi"]:
                
                cur_task_type = raw_turn_id.split("_")[1]
                if prev_task_type != cur_task_type:
                    constraints = []
                    prev_task_type = cur_task_type
                query = item["inst"]
                constraints.append(query)
                
                doc_i = int(data.get("id").split("_")[0]) - 1
                reference_document = self.documents[doc_i]
                eval_config = TurnEvalConfig(
                                do_eval=do_eval,
                                metrics=[MetricConfig(class_name="llm_judge", args={"template_name": "refinement_multi_evaluation", "constraints": constraints})]
                            )
            
            elif task_name in ["refinement_single"]:
                
                cur_task_type = raw_turn_id.split("_")[1]
                if prev_task_type != cur_task_type:
                    constraints = []
                    prev_task_type = cur_task_type
                query = item["inst"]
                constraints.append(query)
                
                
                doc_i = int(raw_turn_id.split("_")[0]) - 1
                reference_document = self.documents[doc_i]
                eval_config = TurnEvalConfig(
                                do_eval=do_eval,
                                metrics=[MetricConfig(class_name="llm_judge", args={"template_name": "refinement_single_evaluation", "constraints": constraints})]
                            )
            
            elif task_name in ["expansion_multi", "expansion_single"]:
                
                constraints = [item["inst"]]
                
                if "multi" in task_name:
                    doc_i = int(raw_turn_id.split("_")[0]) - 1
                else:
                    doc_i = int(raw_turn_id.split("#")[1].split("_")[0]) - 1   
                reference_document = self.documents[doc_i]
                eval_config = TurnEvalConfig(
                                do_eval=do_eval,
                                metrics=[MetricConfig(class_name="llm_judge", args={"template_name": "expansion_evaluation", "constraints": constraints})]
                            )
            
            elif task_name in ["follow-up_multi", "follow-up_single"]:
                
                eval_config = TurnEvalConfig(
                                do_eval=do_eval,
                                metrics=[MetricConfig(class_name="llm_judge", args={"template_name": "mt_bench_evaluation"})]
                            )
                
            elif task_name in ["recollection_multi_cls", "recollection_single_cls"]: # precision for cls tasks
                eval_config = TurnEvalConfig(
                                do_eval=do_eval,
                                metrics=[MetricConfig(class_name="precision")]
                            )
            
            elif task_name in ["recollection_multi_global-inst", "recollection_single_global-inst"]: # 指向相应的指令，并准备好入参
                cur_task_type = data.get("inst_name")

                eval_config = TurnEvalConfig(
                    do_eval=do_eval,
                    metrics=[MetricConfig(class_name="instruction_following", args={"inst_name": data.get("inst_name"), "inst_args": data.get("inst_args")})] 
                )
            
            else:
                warnings.warn(f"Unknown task_name '{task_name}' encountered in processing conversation. No evaluation configuration is set for this task.")
                eval_config = TurnEvalConfig(do_eval=False, metrics=[])
            
            turns.append(
                Turn(
                    turn_id=turn_id,
                    role="assistant",
                    content=assistant_text if not do_eval else None, # 需要测试，则为空；不需要测试，则为参考语句
                    reference=assistant_text,
                    reference_document=reference_document,
                    eval_config=eval_config,
                    turn_labels={"cur_task_type": cur_task_type} if cur_task_type else {},
                )
            )
            turn_id += 1

        
        dialogue_eval_config = DialogEvalConfig(
            use_reference_history=False, # 决定了历史assistant语句是选择reference还是选择content
        )

        return Dialog(dialog_id=dialog_id, 
                      dialog_labels=dialog_labels, 
                      dialog_eval_config=dialogue_eval_config,
                      dialog_turns=turns,
                      dialog_raw_info=dialog_raw_info)


