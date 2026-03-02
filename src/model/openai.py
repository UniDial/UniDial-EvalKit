from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional
import time

from .base import BaseModel

logger = logging.getLogger(__name__)

try:
    import openai
    from openai import OpenAI
except ImportError:
    openai = None
    OpenAI = None


class OpenAIModel(BaseModel):
    """
    OpenAI API Wrapper for chat completion.
    Only supports standard OpenAI API.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        **kwargs: Any
    ) -> None:
        """
        Initialize the OpenAI client.
        
        Args:
            model_name: The name of the model to use (e.g., "gpt-4", "gpt-3.5-turbo").
            api_key: OpenAI API key (optional if env var set).
            base_url: Custom API base URL (optional).
            max_retries: Number of retries for failed requests (default: 3).
            timeout: Request timeout in seconds (default: 60).
            **kwargs: Additional client configuration args.
        """
        super().__init__(model_name, **kwargs)
        
        if OpenAI is None:
            raise ImportError("OpenAI package is missing. Install it via `pip install openai`.")

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            **kwargs
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any
    ) -> str:
        """
        Generate a response using OpenAI ChatCompletion.

        Args:
            messages: A list of message dictionaries.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            top_p: Nucleus sampling parameter.
            n: Number of completions to generate (default 1).
            **kwargs: Additional generation parameters.

        Returns:
            The generated content string.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            if not response.choices:
                raise ValueError("OpenAI API returned no choices in response")
            
            content = response.choices[0].message.content
            if not content:
                try:
                    reasoning_content = response.choices[0].message.reasoning_content
                    if reasoning_content:
                        return reasoning_content
                except Exception as e:
                    raise ValueError("OpenAI API returned empty content in response")
            
            return content
            
        except openai.OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error in OpenAIModel.generate: {e}")
            raise e

