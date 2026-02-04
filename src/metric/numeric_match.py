mport re
from typing import Any, Dict, Optional, Union

from .base import BaseMetric

class NumericMatchMetric(BaseMetric):
    """
    Extracts the last number from the prediction and compares it with the reference number.
    """
    
    def compute(
        self,
        prediction: Union[str, int, float],
        reference: Optional[Union[str, int, float]],
        history_messages: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        
        if reference is None:
            return {"score": 0.0, "rationale": "No reference provided"}

        pred_str = str(prediction)
        ref_str = str(reference)
        
          # Helper to extract last number
        def extract_last_number(text: str) -> Optional[float]:
            # Regex to find numbers (integers or decimals)
            # This regex captures numbers like 123, 123.45, -123.
            # It might need refinement for complex math expressions, but standard for these benchmarks.
            # \d+(?:,\d{3})*(?:\.\d+)? handles commas like 1,000.00
            matches = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
            if not matches:
                return None
            
            last_num_str = matches[-1].replace(',', '')
            try:
                return float(last_num_str)
            except ValueError:
                return None

        pred_num = extract_last_number(pred_str)
        ref_num = extract_last_number(ref_str)
        
        if pred_num is None or ref_num is None:
            # If we can't extract numbers from either, fallback to exact string match?
            # Or strict failure? Usually 0 score if number expected but not found.
            # But let's check if prediction is just text "The answer is 5." -> 5.0
            
            # If extraction failed, maybe try exact match of sanitized strings?
            # For now, if extraction fails, 0.0 unless they are identical strings.
            if pred_str.strip() == ref_str.strip():
                return {"score": 1.0, "match": True}
            
            return {
                "score": 0.0,
                "pred_num": pred_num,
                "ref_num": ref_num,
                "rationale": "Number extraction failed"
            }

        # Compare with tolerance
        tolerance = 1e-6
        is_match = abs(pred_num - ref_num) < tolerance
        
        return {
            "score": 1.0 if is_match else 0.0,
            "match": is_match,
            "pred_num": pred_num,
            "ref_num": ref_num
        }
