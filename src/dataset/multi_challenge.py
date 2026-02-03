from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Any

from .base import BenchmarkContext, BenchmarkDataset
from .data_utils import iter_jsonl_lines
from .schema import Dialog, Turn, TurnEvalConfig, MetricConfig, DialogEvalConfig

logger = logging.getLogger(__name__)


def _iter_multichallenge_files(raw_path: Path) -> Iterator[Path]:
    """Iterate over Multi-Challenge JSONL files."""
    for p in sorted(raw_path.glob("*.jsonl")):
        if not p.name.startswith(".") and not p.name.startswith("_"):
            yield p


class MultiChallengeDataset(BenchmarkDataset):
    benchmark_id = "multi_challenge"

    def prompt_template_render(
        self,
        template_name: str, 
        prediction: str, 
        criteria: str,
    ) -> str:
        """Render the prompt template for Multi-Challenge evaluation."""
        prompt_templates = self.prompt_templates()
        template = prompt_templates.get(template_name)
        if not template:
            raise ValueError(f"Template {template_name} not found")
        
        # Format the template with MODEL_RESPONSE and CRITERIA
        return template.format(response = prediction, criteria = criteria)

    def prompt_templates(self) -> Dict[str, str]:
        """Provide prompt templates for Multi-Challenge evaluation."""
        JUDGE_PROMPT = '''You are tasked with evaluating a model response to see if it meets a specific criteria.
The criteria will always be YES/NO evaluation.

The model response is as follows:
<MODEL_RESPONSE>
{response}
</MODEL_RESPONSE>

The criteria that the model response must meet is as follows. Be VERY STRICT!:
<CRITERIA>
{criteria}
</CRITERIA>

Evaluate the model response where score is 1 if the criteria is met (YES) and 0 if the criteria is not met (NO). Provide your verdict in the following JSON format:
```json
{{
  "Rationale": "<Explain your reasoning for the verdict>",
  "Score": <1 if YES, 0 if NO>,
}}
```
'''
        
        return {
            "multi_challenge_evaluation": JUDGE_PROMPT,
        }

    def metric_configs(self) -> List[str]:
        """Return the list of metric configurations for this dataset."""
        return {"llm_judge": {"min_score": 0, "max_score": 1}}

    def _normalize_raw_data(self, ctx: BenchmarkContext) -> Iterable[Dialog]:
        """Normalize raw Multi-Challenge data into Dialog objects."""
        raw_path = ctx.raw_path
        logger.info(f"MultiChallengeDataset: checking raw_path: {raw_path.absolute()}")
        if not raw_path.exists():
            logger.error(f"MultiChallengeDataset: raw_path does not exist: {raw_path}")
            raise FileNotFoundError(raw_path)

        dialog_index = -1
        for file_path in _iter_multichallenge_files(raw_path):
            for line_index, line in enumerate(iter_jsonl_lines(file_path)):
                if line_index == 2:
                    break
                data = json.loads(line)
                dialog_index += 1
                yield self._build_dialog(
                    data,
                    dialog_id=dialog_index,
                    source_file=file_path.name,
                    line_index=line_index,
                )

    def _build_dialog(
        self,
        data: Dict,
        *,
        dialog_id: int,
        source_file: str,
        line_index: int,
    ) -> Dialog:
        """Build a Dialog object from raw Multi-Challenge data."""
        conversation = data.get("CONVERSATION", [])
        axis = data.get("AXIS", "")
        target_question = data.get("TARGET_QUESTION", "")
        pass_criteria = data.get("PASS_CRITERIA", "")

        dialog_labels: Dict[str, object] = {
            "task_type": axis,
        }

        dialog_raw_info: Dict[str, object] = {
            "question_id": data.get("QUESTION_ID", ""),
            "source_file": source_file,
            "line_index": line_index,
        }

        turns: List[Turn] = []
        turn_id = 0

        # Process conversation turns
        for idx, item in enumerate(conversation):
            role = item.get("role", "")
            content = item.get("content", "")

            if role == "user":
                turns.append(Turn(turn_id=turn_id, role="user", content=content))
                turn_id += 1
            elif role == "assistant":
                
                eval_config = TurnEvalConfig(do_eval=False, metrics=[])

                turns.append(
                    Turn(
                        turn_id=turn_id,
                        role="assistant",
                        content=content,  # If evaluating, content is None (to be generated)
                        eval_config=eval_config,
                        turn_labels={"axis": axis} if axis else {},
                    )
                )
                turn_id += 1

        turns.append(
            Turn(
                turn_id=turn_id,
                role="assistant",
                content = None,
                reference = None,
                eval_config=TurnEvalConfig(do_eval=True, metrics=[MetricConfig(class_name="llm_judge", args={"template_name": "multi_challenge_evaluation", "criteria": target_question})]),
            )
        )
        
        dialogue_eval_config = DialogEvalConfig(
            use_reference_history=True,
        )

        return Dialog(
            dialog_id=dialog_id,
            dialog_labels=dialog_labels,
            dialog_eval_config=dialogue_eval_config,
            dialog_turns=turns,
            dialog_raw_info=dialog_raw_info,
        )

