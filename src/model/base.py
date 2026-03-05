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

    def begin_dialog(self, dialog_id: Optional[int] = None, **kwargs: Any) -> None:
        """
        Optional hook called before generating a dialog.
        Stateful agent models can override this to allocate dialog-scoped resources.
        """
        return None

    def end_dialog(self, dialog_id: Optional[int] = None, **kwargs: Any) -> None:
        """
        Optional hook called after generating a dialog (success or failure).
        Stateful agent models can override this to release dialog-scoped resources.
        """
        return None

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

