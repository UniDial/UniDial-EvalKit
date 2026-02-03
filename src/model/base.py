from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional

class BaseModel(abc.ABC):
    """
    Base class for evaluation objects (Models/Agents).
    Responsible for initializing resources and generating responses based on dialogue history.
    """

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        """
        Initialize the model with resources.
        
        Args:
            model_name: Identifier for the model.
            **kwargs: Configuration parameters for API/Agent/LLM resources.
        """
        self.model_name = model_name
        self.config = kwargs

    @abc.abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """
        Generate a response for the given dialogue history.

        Args:
            messages: A list of message dictionaries. 
                      Standard format: [{"role": "user", "content": "..."}]
            **kwargs: Generation-specific parameters (e.g., temperature, max_tokens) that override defaults.

        Returns:
            The generated text response.
        """
        pass

