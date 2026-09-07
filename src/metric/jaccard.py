from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from src.registry import register_metric

from .base import BaseMetric, coerce_to_sequence


@register_metric("jaccard")
class JaccardMetric(BaseMetric):
    """Jaccard similarity between two sequences.

    - Equal-length sequences: aligned formula ``matched / (2 * n - matched)``
      (same as Amemgym overall).
    - Otherwise: classic set Jaccard ``|A ∩ B| / |A ∪ B|``.

    Inputs are normalized via ``coerce_to_sequence``.

    Optional kwargs:
    - ``lower``: case-fold elements after coercion
    - ``split_special_start_token`` / ``split_special_end_token``: fence strip
    - ``choice_sequences``: map 1-based answer index → sequence
    - ``reference_sequence``: override ``reference`` when provided
    - ``sequence_key``: dict field to extract when coercing objects
    - ``mode``: ``"auto"`` (default), ``"aligned"``, or ``"set"``
    """

    def __init__(self, lower: bool = False, mode: str = "auto"):
        super().__init__()
        self.lower = lower
        self.mode = mode

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
        mode = str(kwargs.get("mode", self.mode) or "auto").lower()
        coerce_kwargs = {
            "lower": lower,
            "split_special_start_token": kwargs.get("split_special_start_token"),
            "split_special_end_token": kwargs.get("split_special_end_token"),
            "choice_sequences": kwargs.get("choice_sequences"),
            "sequence_key": kwargs.get("sequence_key"),
        }

        pred_seq = coerce_to_sequence(prediction, **coerce_kwargs)
        ref_value = kwargs.get("reference_sequence", reference)
        ref_coerce = dict(coerce_kwargs)
        if "reference_sequence" in kwargs:
            ref_coerce["choice_sequences"] = None
        ref_seq = coerce_to_sequence(ref_value, **ref_coerce)

        if not pred_seq and not ref_seq:
            return {
                "score": 1.0,
                "mode": mode,
                "pred_sequence": pred_seq,
                "ref_sequence": ref_seq,
            }

        use_aligned = mode == "aligned" or (
            mode == "auto" and len(pred_seq) == len(ref_seq) and len(pred_seq) > 0
        )
        if mode == "set":
            use_aligned = False

        if use_aligned:
            if len(pred_seq) != len(ref_seq):
                return {
                    "score": 0.0,
                    "rationale": "Aligned Jaccard requires equal-length sequences",
                    "mode": "aligned",
                    "pred_sequence": pred_seq,
                    "ref_sequence": ref_seq,
                }
            matched = sum(a == b for a, b in zip(pred_seq, ref_seq))
            denom = len(pred_seq) * 2 - matched
            score = (matched / denom) if denom else 0.0
            used_mode = "aligned"
        else:
            pred_set = set(pred_seq)
            ref_set = set(ref_seq)
            intersection = pred_set & ref_set
            union = pred_set | ref_set
            score = (len(intersection) / len(union)) if union else 0.0
            matched = len(intersection)
            used_mode = "set"

        return {
            "score": float(score),
            "mode": used_mode,
            "matched": matched,
            "pred_sequence": pred_seq,
            "ref_sequence": ref_seq,
        }
