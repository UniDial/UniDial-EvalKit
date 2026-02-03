from __future__ import annotations

import abc
from typing import Any, Dict, Optional, Union, List
import evaluate



class BaseMetric(abc.ABC):
    """
    Metric 只包含逻辑，不持有数据；数据与运行时信息通过 kwargs 注入。
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

try:
    import evaluate
except Exception:
    evaluate = None


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



class ExactMatchMetric(BaseMetric):
    """
    Exact Match metric.
    1) Try HuggingFace evaluate exact_match
    2) Fallback to word-level exact match
    """

    def __init__(self, lower: bool = True):
        super().__init__()
        self.lower = lower
        self.metric = None

        if evaluate is not None:
            try:
                self.metric = evaluate.load("exact_match")
            except Exception:
                self.metric = None

    def compute(
        self,
        prediction: Union[str, int, float],
        reference: Optional[Union[str, int, float]],
        history_messages: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:

        if reference is None:
            return {"score": 0.0, "rationale": "No reference provided"}

        if self.lower:
            prediction = prediction.lower()
            reference = reference.lower()
            
        if self.metric is not None:
            try:
                result = self.metric.compute(
                    predictions=[prediction],
                    references=[reference],
                )
                score = result.get("exact_match", list(result.values())[0])
                return {"score": score, "exact_match": score}
            except Exception:
                pass
        
        return {"score": 0.0, "error": "ExactMatchMetric failed"}

       

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
            "precision": score,
            "overlap": overlap,
            "pred_len": len(p),
            "ref_len": len(r),
        }


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
            return {
                "score": f1,
                "f1": f1,
            }
        else:
            res = self._get_f1_score(prediction, reference)
            return {
                "score": res["f1"],
                **res
            }

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
