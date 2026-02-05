import ast
import re
from typing import Any, Dict, List, Optional, Union
import numpy as np
from .base import BaseMetric
from .extract_objects_for_code_match import get_all_objects

def extract_code(text: str) -> str:
    """Extracts the Python code from the LM output"""
    # Try multiple code wrappers ```python and ```
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    code_match = pattern.findall(text)
    if not code_match:
        pattern = re.compile(r"```(.*?)```", re.DOTALL)
        code_match = pattern.findall(text)
    extracted_text = code_match[0] if code_match else text
    return extracted_text

class CodeMatchMetric(BaseMetric):
    metric_name: str = "code_match"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute(
        self,
        prediction: str,
        reference: Optional[str] = None,
        history_messages: Dict[str, Any] = None,
        regex: Any = None,
        **kwargs
    ) -> Dict[str, Any]:
        if not prediction:
            return {"score": 0.0, "explanation": "Empty prediction"}

        if not regex:
            return {"score": 0.0, "explanation": "No regex config provided"}
            
        configs_to_eval = []
        if isinstance(regex, list) and len(regex) > 0 and isinstance(regex[0], list):
             configs_to_eval = regex
        else:
             configs_to_eval = [regex]
             
        scores = []
        
        python_text_code = extract_code(prediction)
        
        try:
            all_found_objects = get_all_objects(python_text_code)
        except SyntaxError:
             return {"score": 0.0, "explanation": "SyntaxError in extracted code"}
         

        for config in configs_to_eval:
            if not config or not isinstance(config, list) or len(config) < 2:
                continue

            object_type = config[0]
            rule = config[1]
            
            found_objects = all_found_objects.get(object_type, [])
            # print("found_objects: ", found_objects)
            
            # Absent objects are not penalized -> return NaN (which Aggregator should handle or we treat as ignore)
            if len(found_objects) == 0 and object_type not in ["comment", "import"]:
                scores.append(0.0)
                continue

            score = 0.0
            
            if isinstance(rule, bool):
                # "Always include objects" or "Never include objects"
                is_present = [len(o) > 0 for o in found_objects]
                if rule: # Must be present
                    score = 1.0 if all(is_present) else 0.0
                else: # Must NOT be present
                    score = 1.0 if not any(is_present) else 0.0
                    
            elif isinstance(rule, list):
                name, value = rule
                if object_type not in ["comment", "import"]:
                    is_correct = [(name in o) == value for o in found_objects]
                    score = 1.0 if np.mean(is_correct) == 1 else 0.0
                else:
                    flat_objects = [item for sublist in found_objects for item in sublist]
                    score = 1.0 if (name in flat_objects) == value else 0.0
                    
            else: # Regex string
                matches = [bool(re.match(r"{}".format(rule), obj[0])) for obj in found_objects if len(obj) > 0]
                score = 1.0 if matches and np.mean(matches) == 1 else 0.0
            
            scores.append(score)
            
        final_score = float(np.mean(scores))
        return {"score": final_score, "explanation": f"Evaluated {len(scores)} rules"}

