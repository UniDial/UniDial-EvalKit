from __future__ import annotations

import logging
import os
import tempfile
import uuid
from typing import Any, Dict, List, Optional

from .base import BaseModel

logger = logging.getLogger(__name__)

try:
    from memoryos import Memoryos
except ImportError:
    Memoryos = None


class MemoryOSModel(BaseModel):
    """
    MemoryOS Model Wrapper for chat completion.
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
        Initialize the MemoryOS client.
        
        Args:
            model_name: The name of the LLM model to use (e.g., "deepseek-chat").
            api_key: API key for the LLM (optional if env var set).
            base_url: Custom API base URL (optional).
            max_retries: Number of retries for failed requests (default: 3).
            timeout: Request timeout in seconds (default: 60).
            **kwargs: Additional configuration parameters.
        """
        super().__init__(model_name, **kwargs)
        
        self.raw_model_name = model_name
        if "_" in model_name:
            self.llm_model_name = model_name.split("_", 1)[1]
        else:
            self.llm_model_name = model_name

        if Memoryos is None:
            raise ImportError("memoryos package is missing. Install it or ensure it's in your Python path.")
        
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Since each dialogue represents a unique user and should not share memory,
        # we do NOT initialize the MemoryOS client here in __init__.
        # Instead, we will instantiate it dynamically in generate() or manage
        # a cache of clients mapped by some dialog identifier if provided in messages.
        # However, the standard `generate` signature only takes `messages`.
        # To handle dialog-level isolation based solely on `messages`, we need to 
        # reconstruct the MemoryOS state from the history, or if MemoryOS has to run 
        # sequentially for a dialog, we must feed the entire history.
        
        # Let's keep the config parameters to initialize the client later.
        self.memoryos_config = {
            "short_term_capacity": kwargs.get("short_term_capacity", 10),
            "mid_term_heat_threshold": kwargs.get("mid_term_heat_threshold", 2000),
            "retrieval_queue_capacity": kwargs.get("retrieval_queue_capacity", 7),
            "long_term_knowledge_capacity": kwargs.get("long_term_knowledge_capacity", 100),
            "mid_term_similarity_threshold": kwargs.get("mid_term_similarity_threshold", 0.6),
            "embedding_model_name": kwargs.get("embedding_model_name", "all-MiniLM-L6-v2")
        }

    def generate(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any
    ) -> str:
        """
        Generate a response using MemoryOS.
        Args:
            messages: A list of message dictionaries (history + current query).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional generation parameters.

        Returns:
            The generated content string.
        """
        try:
            if not messages:
                return ""
            
            # Extract final user query
            final_query = ""
            for msg in reversed(messages):
                if msg.get("role", "").lower() == "user":
                    final_query = str(msg.get("content", ""))
                    break
            
            if not final_query:
                logger.warning("No user message found in messages")
                return ""
            # Create a unique temporary MemoryOS client for this generation call.
            session_id = str(uuid.uuid4())[:8]
            temp_dir = tempfile.mkdtemp(prefix=f"memoryos_eval_{session_id}_")
            
            client = Memoryos(
                user_id=f"user_{session_id}",
                openai_api_key=self.api_key,
                openai_base_url=self.base_url,
                data_storage_path=temp_dir,
                llm_model=self.llm_model_name,
                assistant_id=f"assistant_{session_id}",
                **self.memoryos_config
            )

            # 2. Reconstruct memory from history (all messages before the final query)
            # MemoryOS needs paired (user_input, agent_response)
            current_user_msg = None
            
            # The last message is the final_query (if it's a user message)
            # Let's iterate up to the last user message
            last_user_idx = -1
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role", "").lower() == "user":
                    last_user_idx = i
                    break

            for i in range(last_user_idx):
                msg = messages[i]
                role = msg.get("role", "").lower()
                content = str(msg.get("content", ""))
                
                if not content:
                    continue
                    
                if role == "user":
                    current_user_msg = content
                elif role == "assistant" and current_user_msg is not None:
                    # Add history pair to MemoryOS short-term memory
                    client.add_memory(user_input=current_user_msg, agent_response=content)
                    current_user_msg = None

            # 3. Handle special mt_eval format for the first query (if applicable)
            # Some datasets have "Instruction:" embedded in the first turn's context.
            # MemoryOS is a conversational agent without document indices. The safest
            # fallback is to feed the exact text (with context and instruction) directly
            # into the get_response query. MemoryOS will incorporate it into short-term
            # memory, which will later become mid/long-term memory.
            # No special parsing for `Instruction:` is strictly needed for MemoryOS 
            # unless you explicitly want to strip context (which we generally don't, 
            # as it needs it to answer).
            
            # 4. Generate response for the final query
            response = client.get_response(query=final_query)
            
            return response if response else ""
            
        except Exception as e:
            logger.error(f"MemoryOS generation error: {e}")
            raise e


