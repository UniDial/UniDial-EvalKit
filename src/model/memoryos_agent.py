from __future__ import annotations

import logging
import os
import tempfile
import uuid
import threading
from typing import Any, Dict, List, Optional

from .base import BaseModel

logger = logging.getLogger(__name__)

try:
    from .memoryos.memoryos import Memoryos
except ImportError as e:
    logger.error(f"ImportError for memoryos: {e}. ")
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
        Initialize the MemoryOS client wrapper.

        Args:
            model_name: The name of the LLM model to use.
            api_key: API key for the LLM.
            base_url: Custom API base URL.
            max_retries: Number of retries for failed requests.
            timeout: Request timeout in seconds.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(model_name, **kwargs)

        self.raw_model_name = model_name
        if "_" in model_name:
            self.llm_model_name = model_name.split("_", 1)[1]
        else:
            self.llm_model_name = model_name

        if Memoryos is None:
            raise ImportError("memoryos package not found.")

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "")
        self.max_retries = max_retries
        self.timeout = timeout

        # Dialog-scoped state
        self._dialog_states: Dict[int, Dict[str, Any]] = {}
        self._state_lock = threading.Lock()

        # MemoryOS configuration
        self.memoryos_config = {
            "short_term_capacity": kwargs.get("short_term_capacity", 10),
            "mid_term_heat_threshold": kwargs.get("mid_term_heat_threshold", 2000),
            "retrieval_queue_capacity": kwargs.get("retrieval_queue_capacity", 3),
            "long_term_knowledge_capacity": kwargs.get("long_term_knowledge_capacity", 100),
            "mid_term_similarity_threshold": kwargs.get("mid_term_similarity_threshold", 0.6),
            "embedding_model_name": kwargs.get("embedding_model_name", "all-MiniLM-L6-v2"),
        }

    def _create_client(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Create a MemoryOS client.

        """
        session_id = str(uuid.uuid4())[:8]
        temp_dir = tempfile.mkdtemp(prefix=f"memoryos_eval_{session_id}_")

        client = Memoryos(
            user_id=f"user_{session_id}",
            openai_api_key=self.api_key,
            openai_base_url=self.base_url,
            data_storage_path=temp_dir,
            llm_model=self.llm_model_name,
            assistant_id=f"assistant_{session_id}",
            **self.memoryos_config,
        )

        return {
            "client": client,
            "temp_dir": temp_dir,
        }

    def begin_dialog(self, dialog_id: Optional[int] = None, **kwargs: Any) -> None:
        """
        Initialize MemoryOS client for a new dialog.
        """
        if dialog_id is None:
            return

        with self._state_lock:
            if dialog_id not in self._dialog_states:
                client_info = self._create_client()
                self._dialog_states[dialog_id] = {
                    "client": client_info["client"],
                    "last_ingested_idx": 0,
                    "temp_dir": client_info["temp_dir"],
                }

    def end_dialog(self, dialog_id: Optional[int] = None, **kwargs: Any) -> None:
        """
        Cleanup MemoryOS resources.

        """
        if dialog_id is None:
            return

        with self._state_lock:
            self._dialog_states.pop(dialog_id, None)

    def generate(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any
    ) -> str:
        """
        Generate a response using MemoryOS with dialog-scoped state.

        """
        try:
            if not messages:
                return ""

            # Extract final user query
            final_query = ""
            last_user_idx = -1
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role", "").lower() == "user":
                    final_query = str(messages[i].get("content", ""))
                    last_user_idx = i
                    break

            if not final_query:
                logger.warning("No user message found in messages")
                return ""

            dialog_id = kwargs.get("dialog_id")

            # Use dialog-scoped state if dialog_id is provided
            if dialog_id is not None:
                with self._state_lock:
                    state = self._dialog_states.get(dialog_id)
                    if state is None:
                        client_info = self._create_client()
                        state = {
                            "client": client_info["client"],
                            "last_ingested_idx": 0,
                            "temp_dir": client_info["temp_dir"],
                        }
                        self._dialog_states[dialog_id] = state
            else:
                # Stateless fallback: create an ephemeral client
                client_info = self._create_client()
                state = {
                    "client": client_info["client"],
                    "last_ingested_idx": 0,
                    "temp_dir": client_info["temp_dir"],
                }

            client = state["client"]
            start_idx = state["last_ingested_idx"]

            # Ingest newly completed history pairs from [start_idx, last_user_idx)
            current_user_msg = None
            i = start_idx
            while i < last_user_idx:
                msg = messages[i]
                role = msg.get("role", "").lower()
                content = msg.get("content")

                if content is None:
                    i += 1
                    continue

                content = str(content).strip()
                if not content:
                    i += 1
                    continue

                if role == "user":
                    current_user_msg = content
                elif role == "assistant" and current_user_msg is not None:
                    client.add_memory(user_input=current_user_msg, agent_response=content)
                    current_user_msg = None

                i += 1

            # Update state cursor
            if dialog_id is not None:
                self._dialog_states[dialog_id]["last_ingested_idx"] = len(messages)

            # Generate response for current query
            response = client.get_response(query=final_query)

            return response if response else ""

        except Exception:
            logger.exception("MemoryOS generation error")
            raise