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
        save_llm_logs: bool = False,
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
        self.save_llm_logs = save_llm_logs
        
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
        # time.sleep(15)
        # Do not forward unset optional params (e.g., temperature=None) to the backend.
        # This lets the backend use its own defaults when the user didn't specify.
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        print(kwargs)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs
            )
            # print(response)
            # exit(0)
            if not response.choices:
                raise ValueError("OpenAI API returned no choices in response")
            
            content = response.choices[0].message.content
            if not content:
                try:
                    msg = response.choices[0].message
                    # Try different possible attribute names for reasoning
                    reasoning_content = getattr(msg, "reasoning_content", None)
                    if not reasoning_content:
                        reasoning_content = getattr(msg, "reasoning", None)
                    
                    # Also check for reasoning_details
                    if not reasoning_content:
                        reasoning_details = getattr(msg, "reasoning_details", None)
                        if isinstance(reasoning_details, list) and len(reasoning_details) > 0:
                            reasoning_content = reasoning_details[0].get("text", "")
                            
                    # Pydantic models might require accessing the dict directly 
                    # if the attribute is not defined in the model schema
                    if not reasoning_content and hasattr(msg, "model_extra") and msg.model_extra:
                        print("here:!!! model_extra")
                        reasoning_content = msg.model_extra.get("reasoning") or msg.model_extra.get("reasoning_content")
                        if not reasoning_content and "reasoning_details" in msg.model_extra:
                            rd = msg.model_extra["reasoning_details"]
                            if isinstance(rd, list) and len(rd) > 0:
                                reasoning_content = rd[0].get("text", "")
                                
                    if reasoning_content:
                        # print(reasoning_content)
                        if self.save_llm_logs:
                            return reasoning_content, response
                        else:
                            return reasoning_content
                except Exception as e:
                    pass
                raise ValueError("OpenAI API returned empty content in response")
            
            if self.save_llm_logs:
                return content, response
            else:
                return content
            
        except openai.OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error in OpenAIModel.generate: {e}")
            raise e     
