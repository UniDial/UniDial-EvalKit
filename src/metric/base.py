from __future__ import annotations

import abc
import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Union

try:
    import evaluate
except Exception:
    evaluate = None

from src.registry import register_metric


class BaseMetric(abc.ABC):
    """
    Metric contains only logic, not data; data and runtime info are injected via kwargs.
    """
    @abc.abstractmethod
    def compute(
        self,
        prediction: str,
        reference: Optional[str],
        history_messages: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        raise NotImplementedError

import re
from collections import Counter
from typing import Any, Dict, Optional, Union

try:
    from nltk.stem import PorterStemmer
except ImportError:
    PorterStemmer = None


# -------------------------
# Shared utilities
# -------------------------

def _word_tokenize(text: str, lower: bool = True):
    if text is None:
        return []
    s = str(text).strip()
    if lower:
        s = s.lower()
    return re.findall(r"\w+", s, flags=re.UNICODE)


def _multiset_overlap(a, b):
    return sum((Counter(a) & Counter(b)).values())


def _strip_special_tokens(
    text: str,
    *,
    start_token: Optional[str] = None,
    end_token: Optional[str] = None,
) -> str:
    out = str(text).strip()
    if start_token:
        out = out.split(start_token)[-1].strip()
        if end_token and out.endswith(end_token):
            out = out[: -len(end_token)].strip()
    return out


def _parse_index(value: Any) -> Optional[int]:
    """Parse a 1-based index from int/float/str/JSON ``{"answer": i}``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, dict) and "answer" in value:
        try:
            return int(value["answer"])
        except (TypeError, ValueError):
            return None
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        payload = None

    if isinstance(payload, (int, float)) and not isinstance(payload, bool):
        return int(payload)
    if isinstance(payload, dict) and "answer" in payload:
        try:
            return int(payload["answer"])
        except (TypeError, ValueError):
            return None

    match = re.search(r'"answer"\s*:\s*(-?\d+)', text)
    if match:
        return int(match.group(1))
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    return None


def coerce_to_sequence(
    value: Any,
    *,
    lower: bool = False,
    split_special_start_token: Optional[str] = None,
    split_special_end_token: Optional[str] = None,
    choice_sequences: Optional[List[Any]] = None,
    sequence_key: Optional[str] = None,
) -> List[Any]:
    """Normalize heterogeneous metric inputs into a flat sequence.

    Supported forms:
    - ``list`` / ``tuple``: used as-is
    - ``set`` / ``frozenset``: sorted for stability
    - scalar ``int`` / ``float`` / ``bool``: single-element list
    - ``dict``: optional ``sequence_key`` value, else ``state`` / ``sequence`` /
      ``items`` / ``values``; with ``choice_sequences``, ``answer`` index is resolved
    - ``str``: strip optional fences, JSON-decode when possible, else comma-split
      or word-tokenize; with ``choice_sequences``, answer-index JSON is resolved
    """
    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        seq = list(value)
    elif isinstance(value, (set, frozenset)):
        seq = sorted(value, key=lambda x: str(x))
    elif isinstance(value, bool) or isinstance(value, (int, float)):
        if choice_sequences is not None:
            idx = _parse_index(value)
            if idx is not None:
                zero_based = idx - 1
                if 0 <= zero_based < len(choice_sequences):
                    return coerce_to_sequence(
                        choice_sequences[zero_based],
                        lower=lower,
                        sequence_key=sequence_key,
                    )
                return []
        seq = [value]
    elif isinstance(value, dict):
        if choice_sequences is not None:
            idx = _parse_index(value)
            if idx is not None:
                zero_based = idx - 1
                if 0 <= zero_based < len(choice_sequences):
                    return coerce_to_sequence(
                        choice_sequences[zero_based],
                        lower=lower,
                        sequence_key=sequence_key,
                    )
                return []
        key = sequence_key
        if key is None:
            for candidate in ("state", "sequence", "items", "values"):
                if candidate in value:
                    key = candidate
                    break
        if key is not None and key in value:
            return coerce_to_sequence(
                value[key],
                lower=lower,
                choice_sequences=choice_sequences,
                sequence_key=sequence_key,
            )
        seq = list(value.values())
    elif isinstance(value, str):
        text = _strip_special_tokens(
            value,
            start_token=split_special_start_token,
            end_token=split_special_end_token,
        )
        if choice_sequences is not None:
            idx = _parse_index(text)
            if idx is not None:
                zero_based = idx - 1
                if 0 <= zero_based < len(choice_sequences):
                    return coerce_to_sequence(
                        choice_sequences[zero_based],
                        lower=lower,
                        sequence_key=sequence_key,
                    )
                return []
        try:
            payload = json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = None
        if payload is not None and not isinstance(payload, str):
            return coerce_to_sequence(
                payload,
                lower=lower,
                choice_sequences=choice_sequences,
                sequence_key=sequence_key,
            )
        if "," in text:
            seq = [part.strip() for part in text.split(",") if part.strip()]
        else:
            seq = _word_tokenize(text, lower=False)
    else:
        seq = [value]

    if lower:
        normalized = []
        for item in seq:
            if isinstance(item, str):
                normalized.append(item.lower())
            else:
                normalized.append(str(item).lower())
        return normalized
    return seq


@register_metric("exact_match")
class ExactMatchMetric(BaseMetric):
    """
    Exact Match metric.
    1) Try HuggingFace evaluate exact_match
    2) Fallback to word-level exact match
    """

    def __init__(self, lower: bool = True):
        super().__init__()
        self.lower = lower


    def compute(
        self,
        prediction: Union[str, int, float],
        reference: Optional[Union[str, int, float]],
        history_messages: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:

        if reference is None:
            return {"score": 0.0, "rationale": "No reference provided"}

        prediction = str(prediction)
        reference = str(reference)

        start_token = kwargs.get("split_special_start_token")
        end_token = kwargs.get("split_special_end_token")
        if start_token:
            prediction = prediction.split(start_token)[-1].strip()
            if end_token and prediction.endswith(end_token):
                prediction = prediction[:-len(end_token)].strip()

        if self.lower:
            prediction = prediction.lower()
            reference = reference.lower()

        score = 1.0 if prediction == reference else 0.0
        return {"score": score}

       

@register_metric("precision")
class PrecisionMetric(BaseMetric):
    """
    Word-level Precision:
    overlap / len(pred_tokens)
    """

    def __init__(self, lower: bool = True):
        super().__init__()
        self.lower = lower

    def compute(
        self,
        prediction: Union[str, int, float],
        reference: Optional[Union[str, int, float]],
        history_messages: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:

        if reference is None:
            return {"score": 0.0, "rationale": "No reference provided"}

        p = _word_tokenize(prediction, self.lower)
        r = _word_tokenize(reference, self.lower)

        overlap = _multiset_overlap(p, r)
        score = overlap / len(p) if len(p) > 0 else 0.0

        return {
            "score": score,
            "overlap": overlap,
            "pred_len": len(p),
            "ref_len": len(r),
        }


@register_metric("recall")
class RecallMetric(BaseMetric):
    """
    Word-level Recall:
    overlap / len(ref_tokens)
    """

    def __init__(self, lower: bool = True, binary: bool = False):
        super().__init__()
        self.lower = lower
        self.binary = binary

    def _get_recall(self, prediction: str, reference: str) -> float:
        p = _word_tokenize(prediction, self.lower)
        r = _word_tokenize(reference, self.lower)
        overlap = _multiset_overlap(p, r)
        return overlap / len(r) if len(r) > 0 else 0.0

    def compute(
        self,
        prediction: Union[str, int, float],
        reference: Optional[Union[str, int, float, List[str]]],
        history_messages: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:

        if reference is None:
            return {"score": 0.0, "rationale": "No reference provided"}

        prediction_str = str(prediction)
        if isinstance(reference, list):
            references = [str(r) for r in reference]
        else:
            references = [str(reference)]

        scores = [self._get_recall(prediction_str, r) for r in references]
        max_recall = max(scores) if scores else 0.0

        if self.binary:
            final_score = 1.0 if max_recall >= 1.0 else 0.0
        else:
            final_score = max_recall

        return {
            "score": final_score,
            "recall": scores,
        }

@register_metric("f1_score")
class F1Metric(BaseMetric):
    """
    Word-level F1:
    2 * (precision * recall) / (precision + recall)
    """

    def __init__(self, lower: bool = True, stem: bool = False, multi_answer: bool = False):
        super().__init__()
        self.lower = lower
        self.stem = stem
        self.multi_answer = multi_answer
        self.stemmer = PorterStemmer() if (stem and PorterStemmer is not None) else None

    def _get_f1_score(self, prediction: str, reference: str) -> Dict[str, float]:
        p_tokens = _word_tokenize(prediction, self.lower)
        r_tokens = _word_tokenize(reference, self.lower)

        if self.stemmer:
            p_tokens = [self.stemmer.stem(w) for w in p_tokens]
            r_tokens = [self.stemmer.stem(w) for w in r_tokens]

        overlap = _multiset_overlap(p_tokens, r_tokens)
        
        precision = overlap / len(p_tokens) if len(p_tokens) > 0 else 0.0
        recall = overlap / len(r_tokens) if len(r_tokens) > 0 else 0.0
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "overlap": overlap,
            "pred_len": len(p_tokens),
            "ref_len": len(r_tokens),
        }

    def compute(
        self,
        prediction: Union[str, int, float],
        reference: Optional[Union[str, int, float]],
        history_messages: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:

        if reference is None:
            return {"score": 0.0, "rationale": "No reference provided"}

        prediction = str(prediction)
        reference = str(reference)

        if self.multi_answer:
            predictions = [p.strip() for p in prediction.split(',')]
            references = [r.strip() for r in reference.split(',')]
            
            # Follow locomo-main evaluation: mean of maxes for each reference
            scores = []
            for ref in references:
                if not ref: continue
                # We only need the f1 score for the aggregation
                max_score = max([self._get_f1_score(pred, ref)["f1"] for pred in predictions]) if predictions else 0.0
                scores.append(max_score)
            
            f1 = sum(scores) / len(scores) if scores else 0.0
            return {"score": f1}
        else:
            res = self._get_f1_score(prediction, reference)
            score = res.pop("f1")
            return {"score": score, **res}

# class HuggingFaceMetric(BaseMetric):
#     """
#     A unified wrapper for HuggingFace evaluate metrics.
#     All metric logic is consolidated here.
#     """

#     def __init__(self, metric_name: str, **kwargs):
#         """
#         Initialize the HuggingFace metric.
        
#         Args:
#             **kwargs: Additional arguments passed to evaluate.load().
#         Note:
#             Subclasses must set self.metric_name before calling super().__init__().
#         """
#         super().__init__()
#         self.metric_name = metric_name
        
#         if evaluate is None:
#             # print("here1")
#             self.metric = None
#             print(f"Warning: 'evaluate' library not installed. {self.metric_name} will not work.")
#         else:
#             # try:
#             # print("here2")
#             self.metric = evaluate.load(self.metric_name, **kwargs)
#             # print(self.metric.compute(predictions=[0,1], references=[0,1]))
#             # except Exception as e:
#             #     print(f"Error loading metric '{self.metric_name}': {e}")
#             #     self.metric = None

#     def compute(
#         self,
#         prediction: Union[str, int, float],
#         reference: Optional[Union[str, int, float]],
#         history_messages: Dict[str, Any],
#         **kwargs: Any,
#     ) -> Dict[str, Any]:
#         if self.metric is None:
#             return {"score": 0.0, "error": "Metric not loaded"}

#         if reference is None:
#             return {"score": 0.0, "rationale": "No reference provided"}

#         try:
#             # evaluate expects lists for predictions and references
#             # kwargs are passed to compute() (e.g., average='micro', smooth=True)
#             results = self.metric.compute(predictions=[prediction], references=[reference], **kwargs)
            
#             # Extract the primary score for the 'score' field
#             # Different metrics return different keys (e.g., 'precision', 'exact_match', 'f1')
#             score = 0.0
            
#             # Heuristic to find the main score
#             # 1. Try key matching metric_name
#             if self.metric_name in results:
#                 score = results[self.metric_name]
#             # 2. Try common score keys
#             elif "score" in results:
#                 score = results["score"]
#             elif "accuracy" in results:
#                 score = results["accuracy"]
#             elif "f1" in results:
#                 score = results["f1"]
#             # 3. If only one result, use it
#             elif len(results) == 1:
#                 score = list(results.values())[0]
            
#             output = results.copy()
#             if "score" not in output:
#                 output["score"] = score
            
#             return output

#         except Exception as e:
#             return {"score": 0.0, "error": str(e)}


# class PrecisionMetric(HuggingFaceMetric):
#     def __init__(self, **kwargs):
#         super().__init__(metric_name="precision", **kwargs)


# class RecallMetric(HuggingFaceMetric):
#     def __init__(self, **kwargs):
#         super().__init__(metric_name="recall", **kwargs)


# class ExactMatchMetric(HuggingFaceMetric):
#     def __init__(self, **kwargs):
#         super().__init__(metric_name="exact_match", **kwargs)
