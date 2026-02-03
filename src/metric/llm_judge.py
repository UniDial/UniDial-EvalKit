from __future__ import annotations

import json
import logging
import re
import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from .base import BaseMetric
from src.model.base import BaseModel
from src.dataset.base import BenchmarkDataset

logger = logging.getLogger(__name__)


class LLMJudge(BaseMetric):
    """
    LLM-based Judge metric.
    """

    def __init__(self, llm_client: BaseModel, dataset: BenchmarkDataset, min_score: int = 0, max_score: int = 10):
        """
        Args:
            llm_client: An initialized LLM client instance adhering to BaseModel interface.
        """
        self.llm_client = llm_client
        self.dataset = dataset
        self.min_score = min_score
        self.max_score = max_score
        
    def compute(
        self,
        prediction: str,
        reference: Optional[str],
        history_messages: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Args:
            prediction: The model's response to be evaluated.
            reference: The reference answer or document content.
            history_messages: Context containing "messages" (history).
            **kwargs: Arguments defined in MetricConfig (e.g., template_name, constraints) AND "dataset".
        """
        # 1. Get Dataset and Template Name
        # dataset = kwargs.get("dataset")
        # if not dataset:
        #     return {"score": 0.0, "error": "Dataset instance missing in kwargs"}
        
        # 2. Check required arguments for prompt_template_render
        # Use introspection to find what arguments prompt_template_render needs
        render_method = getattr(self.dataset, "prompt_template_render", None)
        if not render_method or not callable(render_method):
             return {"score": 0.0, "error": f"Dataset {type(dataset).__name__} does not have a callable 'prompt_template_render' method."}

        sig = inspect.signature(render_method)
        
        # Prepare available arguments
        # We explicitly map standard compute args to potential render args
        available_args = {
            "history_messages": history_messages,
            "response": prediction,
            "prediction": prediction, # Alias
            "reference": reference,
            **kwargs # Includes constraints, template_name, etc.
        }
        
        render_kwargs = {}
        missing_args = []
        for param_name, param in sig.parameters.items():
            # Skip self and variable keyword arguments like **kwargs if they exist in signature
            if param_name == "self" or param.kind == inspect.Parameter.VAR_KEYWORD:
                continue
            
            if param_name in available_args:
                render_kwargs[param_name] = available_args[param_name]
            elif param.default == inspect.Parameter.empty:
                missing_args.append(param_name)
        
        if missing_args:
             return {"score": 0.0, "error": f"Missing required arguments for prompt_template_render: {missing_args}"}

        try:
            # 3. Render Prompt (Delegated to Dataset)
            prompt = self.dataset.prompt_template_render(**render_kwargs)
            
            # Ensure prompt is in message format for chat models
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = prompt
            
            # 4. Call LLM
            llm_output = self.llm_client.generate(messages=messages) # TODO: , **kwargs)
            
            # 5. Parse Output
            parsed_result = self._parse_json_output(llm_output)
            
            # 6. normalize score
            parsed_result["score"] = (parsed_result["score"] - self.min_score) / (self.max_score - self.min_score)
            
            return parsed_result

        except Exception as e:
            logger.error(f"LLMJudge failed: {e}")
            return {
                "score": 0.0,
                "error": str(e),
                "raw_output": locals().get("llm_output", "")
            }

    def _parse_json_output(self, text: str) -> Dict[str, Any]:
        """
        Parses JSON output from LLM, handling Markdown code blocks.
        Expected format: {"Score": int, "Rationale": str}
        """
        try:
            # Strip markdown code blocks if present
            text = text.strip()
            if text.startswith("```"):
                # Remove first line (```json or ```)
                text = re.sub(r"^```\w*\n", "", text)
                # Remove last line (```)
                text = re.sub(r"\n```$", "", text)
            
            data = json.loads(text)
            
            # Normalize keys (mt_eval templates use "Score" and "Rationale")
            score = data.get("Score", data.get("score", 0))
            rationale = data.get("Rationale", data.get("rationale", ""))
            
            return {
                "score": float(score),
                "rationale": rationale,
                "raw_output": text
            }
        except json.JSONDecodeError:
            # Fallback: Try to find score using regex if JSON fails
            score_match = re.search(r'"Score":\s*(\d+(\.\d+)?)', text, re.IGNORECASE)
            if score_match:
                return {
                    "score": float(score_match.group(1)),
                    "rationale": "Parsed via regex (invalid JSON)",
                    "raw_output": text
                }
            return {
                "score": 0.0,
                "error": "Failed to parse JSON",
                "raw_output": text
            }
