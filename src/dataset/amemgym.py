from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.registry import register_dataset

from .base import BenchmarkContext, BenchmarkDataset
from .schema import Dialog, DialogEvalConfig, MetricConfig, Turn, TurnEvalConfig


OVERALL_PROMPT = """\
{query}

Please select the most suitable answer for my current situation from the following options:
(considering my current relevant preferences and state information)

{choices}

Express your choice with a number and output in the following JSON format:
```json
{{
    "answer": int
}}
```
Only keep the JSON format output, do not include any other content.
"""


# Copied from amemgym.env.sample_interactions._generate_user_followup
USER_SIMULATOR_PROMPT = """
You are simulating a user in a conversation with an AI assistant. You must continue the conversation - early stopping is not allowed.

Initial User Profile on ({start_date}):
{user_profile}

Current Date: {current_date}

Initial Query: {query}

Recent Conversation (including the latest assistant response):
{context}

Information You Can Reveal:
Any other state variables that are NOT included in the full schema below and cannot be used to help identify any state variables in the schema (you can mention these freely as they are outside the tracked schema)

Full Schema (DO NOT reveal values for variables in this schema):
{state_schema}

Instructions:
1. You MUST continue the conversation - do not end it
2. If the assistant asked for clarification, provide a helpful response using information you can reveal as specified above
    - Don't provide further personal information if not asked
    - Don't repeat information already provided in the initial query
3. If your initial query seems addressed, ask a relevant follow-up question that naturally extends the conversation
4. Consider asking about related topics, implementation details, alternatives, or seeking clarification on specific points
5. Keep responses conversational and natural to your persona
6. You can mention any state variables that are NOT in the schema above, but ensure they cannot help identify values of variables in the schema
    - DO NOT reveal specific values for any state variables that are in the schema
7. Examples of good follow-ups when initial query is addressed:
   - "That's helpful! Could you also tell me about..."
   - "Thanks for that information. I'm also curious about..."
   - "That makes sense. What about..."
   - "Good to know. Is there anything else I should consider regarding..."

You must respond with a natural follow-up response that continues the conversation. Return only the response text, no additional formatting or explanation.
"""


@register_dataset()
class AmemgymDataset(BenchmarkDataset):
    benchmark_id = "amemgym"

    # Matches amemgym configs/env/v1.base.json: max_rounds means user-assistant
    # pairs, and the first user message is the fixed session query.
    DEFAULT_NUM_ROUNDS_INIT = 1 # 1
    DEFAULT_NUM_ROUNDS_UPDATE = 2 #2

    def metric_configs(self) -> Dict[str, Any]:
        return {
            "exact_match": {},
            "hamming": {},
            "jaccard": {},
        }

    @classmethod
    def meta_version(cls) -> int:
        return 6

    def get_default_user_simulator_calls(self, turn: Turn) -> int:
        period_idx = int(turn.turn_labels.get("period_index", 0) or 0)
        max_rounds = self.DEFAULT_NUM_ROUNDS_INIT if period_idx == 0 else self.DEFAULT_NUM_ROUNDS_UPDATE
        return max(max_rounds - 1, 0)

    def render_user_simulator_prompt(
        self,
        *,
        dialog: Dialog,
        history_messages: List[Dict[str, str]],
        current_turn: Turn,
    ) -> str:
        """Render the official Amemgym user-followup prompt."""
        labels = current_turn.turn_labels
        profile = dialog.dialog_raw_info.get("user_profile_formatted") or ""
        # Official sample_interactions uses conversation_history[-2:].
        context = "\n".join(
            f"{msg.get('role', '').title()}: {msg.get('content', '')}"
            for msg in history_messages[-2:]
        )
        # Official Initial Query is the session opener (with [Current Time: ...]).
        query = current_turn.content or labels.get("session_query") or ""
        state_schema = dialog.dialog_raw_info.get("state_schema", {})

        return USER_SIMULATOR_PROMPT.format(
            start_date=dialog.dialog_raw_info.get("start_time", ""),
            user_profile=profile,
            current_date=labels.get("period_end", ""),
            query=query,
            context=context,
            state_schema=json.dumps(state_schema, indent=2, ensure_ascii=False),
        )

    def _normalize_raw_data(self, ctx: BenchmarkContext) -> Iterable[Dialog]:
        data_file = self._resolve_data_file(ctx.raw_path)
        with data_file.open("r", encoding="utf-8") as f:
            samples = json.load(f)

        dialog_id = 0
        for sample_idx, item in enumerate(samples):
            periods = item.get("periods", [])
            qas = item.get("qas", [])
            turns: List[Turn] = []
            turn_id = 0

            for period_idx, period in enumerate(periods):
                turn_id = self._append_environment_session_turns(
                    turns=turns,
                    period=period,
                    period_idx=period_idx,
                    turn_id=turn_id,
                )
                for qa_idx, qa in enumerate(qas):
                    turn_id = self._append_overall_eval_turns(
                        turns=turns,
                        period=period,
                        period_idx=period_idx,
                        qa=qa,
                        qa_idx=qa_idx,
                        turn_id=turn_id,
                    )

            yield Dialog(
                dialog_id=dialog_id,
                dialog_labels={
                    "sample_id": item.get("id"),
                    "task_type": "overall",
                    "num_periods": len(periods),
                    "num_qas": len(qas),
                },
                dialog_raw_info={
                    "source_file": str(data_file),
                    "line_index": sample_idx,
                    "raw_id": item.get("id"),
                    "start_time": item.get("start_time"),
                    "user_profile": item.get("user_profile", {}),
                    "user_profile_formatted": (item.get("user_profile") or {}).get("formatted_str", ""),
                    "state_schema": item.get("state_schema", {}),
                },
                dialog_eval_config=DialogEvalConfig(
                    use_reference_history=False,
                    enable_user_simulator=True,
                ),
                dialog_turns=turns,
            )
            dialog_id += 1

    def _append_environment_session_turns(
        self,
        *,
        turns: List[Turn],
        period: Dict[str, Any],
        period_idx: int,
        turn_id: int,
    ) -> int:
        for session_idx, session in enumerate(period.get("sessions", [])):
            session_query = session.get("query", "")
            query_with_time = f"[Current Time: {session.get('session_time', '')}]\n{session_query}"
            turns.append(
                Turn(
                    turn_id=turn_id,
                    role="user",
                    content=query_with_time,
                    eval_config=TurnEvalConfig(
                        do_eval=False,
                        user_simulator_calls=-1,
                    ),
                    turn_labels={
                        "raw_turn_type": "environment_session_query",
                        "period_index": period_idx,
                        "session_index": session_idx,
                        "period_start": period.get("period_start"),
                        "period_end": period.get("period_end"),
                        "period_summary": period.get("period_summary"),
                        "session_time": session.get("session_time"),
                        "session_query": session_query,
                        "event": session.get("event"),
                        "exposed_states": session.get("exposed_states", {}),
                    },
                )
            )
            turn_id += 1

            turns.append(
                Turn(
                    turn_id=turn_id,
                    role="assistant",
                    content=None,
                    eval_config=TurnEvalConfig(do_eval=False),
                    turn_labels={
                        "raw_turn_type": "environment_session_response",
                        "period_index": period_idx,
                        "session_index": session_idx,
                    },
                )
            )
            turn_id += 1

        return turn_id

    def _append_overall_eval_turns(
        self,
        *,
        turns: List[Turn],
        period: Dict[str, Any],
        period_idx: int,
        qa: Dict[str, Any],
        qa_idx: int,
        turn_id: int,
    ) -> int:
        required_info = qa.get("required_info", [])
        turns.append(
            Turn(
                turn_id=turn_id,
                role="user",
                content=self._format_overall_prompt(qa),
                turn_labels={
                    "raw_turn_type": "overall_question",
                    "raw_include_in_history": False,
                    "period_index": period_idx,
                    "raw_qa_index": qa_idx,
                    "raw_required_info": required_info,
                },
            )
        )
        turn_id += 1

        answer_index = self._golden_answer_index(period, qa)
        answer_choice = qa["answer_choices"][answer_index]
        turns.append(
            Turn(
                turn_id=turn_id,
                role="assistant",
                content=None,
                reference=self._format_answer_reference(answer_index),
                eval_config=TurnEvalConfig(
                    do_eval=True,
                    metrics=self._overall_metric_configs(qa, answer_choice),
                ),
                turn_labels={
                    "raw_turn_type": "overall_answer",
                    "raw_include_in_history": False,
                    "period_index": period_idx,
                    "raw_qa_index": qa_idx,
                    "raw_answer_index": answer_index,
                    "raw_required_info": required_info,
                    "raw_answer_state": answer_choice.get("state"),
                    "raw_answer_choice": answer_choice.get("answer"),
                },
            )
        )
        return turn_id + 1

    @staticmethod
    def _overall_metric_configs(qa: Dict[str, Any], answer_choice: Dict[str, Any]) -> List[MetricConfig]:
        fence_args = {
            "split_special_start_token": "```json",
            "split_special_end_token": "```",
        }
        sequence_args = {
            **fence_args,
            "reference_sequence": answer_choice.get("state"),
            "choice_sequences": [choice.get("state") for choice in qa.get("answer_choices", [])],
            "mode": "aligned",
        }
        return [
            MetricConfig(class_name="exact_match", args=dict(fence_args)),
            MetricConfig(class_name="hamming", args={k: v for k, v in sequence_args.items() if k != "mode"}),
            MetricConfig(class_name="jaccard", args=dict(sequence_args)),
        ]

    @staticmethod
    def _resolve_data_file(raw_path: Path) -> Path:
        data_file = raw_path if raw_path.is_file() else raw_path / "data.json"
        if not data_file.exists():
            raise FileNotFoundError(f"Amemgym data file not found at {data_file}")
        return data_file

    @staticmethod
    def _format_overall_prompt(qa: Dict[str, Any]) -> str:
        choices_text = "\n".join(
            "{}: {}".format(choice_index + 1, choice.get("answer", ""))
            for choice_index, choice in enumerate(qa.get("answer_choices", []))
        )
        return OVERALL_PROMPT.format(query=qa.get("query", ""), choices=choices_text)

    @staticmethod
    def _golden_answer_index(period: Dict[str, Any], qa: Dict[str, Any]) -> int:
        golden_state = [period.get("state", {}).get(info_type) for info_type in qa.get("required_info", [])]
        for choice_index, choice in enumerate(qa.get("answer_choices", [])):
            if choice.get("state") == golden_state:
                return choice_index
        raise ValueError(f"No golden answer choice found for required state {golden_state}")

    @staticmethod
    def _format_answer_reference(answer_index: int) -> str:
        return json.dumps({"answer": answer_index + 1}, indent=4)
