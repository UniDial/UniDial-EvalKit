from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from src.registry import register_metric

from .base import BaseMetric, coerce_to_sequence


@register_metric("hamming")
class HammingMetric(BaseMetric):
    """Position-wise match rate between two sequences.

    ``score = matched / max(len(pred), len(ref))`` (missing positions count as
    mismatches). Inputs are normalized via ``coerce_to_sequence``.

    Optional kwargs:
    - ``lower``: case-fold elements after coercion
    - ``split_special_start_token`` / ``split_special_end_token``: fence strip
    - ``choice_sequences``: map 1-based answer index → sequence
    - ``reference_sequence``: override ``reference`` when provided
    - ``sequence_key``: dict field to extract when coercing objects
    """

    def __init__(self, lower: bool = False):
        super().__init__()
        self.lower = lower

    def compute(
        self,
        prediction: Union[str, int, float, List[Any]],
        reference: Optional[Union[str, int, float, List[Any]]],
        history_messages: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if reference is None and kwargs.get("reference_sequence") is None:
            return {"score": 0.0, "rationale": "No reference provided"}

        lower = bool(kwargs.get("lower", self.lower))
        coerce_kwargs = {
            "lower": lower,
            "split_special_start_token": kwargs.get("split_special_start_token"),
            "split_special_end_token": kwargs.get("split_special_end_token"),
            "choice_sequences": kwargs.get("choice_sequences"),
            "sequence_key": kwargs.get("sequence_key"),
        }

        pred_seq = coerce_to_sequence(prediction, **coerce_kwargs)
        ref_value = kwargs.get("reference_sequence", reference)
        # Reference side usually should not go through choice_sequences unless
        # the caller intentionally stores an answer index as reference.
        ref_coerce = dict(coerce_kwargs)
        if "reference_sequence" in kwargs:
            ref_coerce["choice_sequences"] = None
        ref_seq = coerce_to_sequence(ref_value, **ref_coerce)

        length = max(len(pred_seq), len(ref_seq))
        if length == 0:
            return {
                "score": 1.0,
                "matched": 0,
                "pred_sequence": pred_seq,
                "ref_sequence": ref_seq,
            }

        matched = sum(
            i < len(pred_seq) and i < len(ref_seq) and pred_seq[i] == ref_seq[i]
            for i in range(length)
        )
        score = matched / length
        return {
            "score": float(score),
            "matched": matched,
            "length": length,
            "pred_sequence": pred_seq,
            "ref_sequence": ref_seq,
        }
