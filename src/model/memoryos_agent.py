from __future__ import annotations

import logging
import os
import sys
import tempfile
import uuid
import threading
import time
import json
import re
from typing import Any, Dict, List, Optional

from .base import BaseModel

# Create thread-local storage for log capturing
_thread_local = threading.local()

class ThreadLocalLogHandler(logging.Handler):
    def emit(self, record):
        try:
            # Check if we are currently capturing for a dialog
            if hasattr(_thread_local, 'log_capture_list') and _thread_local.log_capture_list is not None:
                # To prevent infinite recursion if format/emit causes logging
                if not getattr(_thread_local, 'is_emitting', False):
                    _thread_local.is_emitting = True
                    try:
                        msg = self.format(record)
                        _thread_local.log_capture_list.append({
                            "time": time.time(),
                            "level": record.levelname,
                            "logger": record.name,
                            "message": msg
                        })
                    finally:
                        _thread_local.is_emitting = False
        except Exception:
            self.handleError(record)

# Initialize the handler once
_capture_handler = ThreadLocalLogHandler()
_capture_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Set up basic config first (with force=True to ensure console output is enabled)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
# Then add our capture handler
logging.getLogger().addHandler(_capture_handler)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    from memoryos.memoryos import Memoryos
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
        max_retries: int = 10,
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
            "short_term_capacity": kwargs.get("short_term_capacity", 7),
            "mid_term_heat_threshold": kwargs.get("mid_term_heat_threshold", 5),
            "retrieval_queue_capacity": kwargs.get("retrieval_queue_capacity", 3),
            "long_term_knowledge_capacity": kwargs.get("long_term_knowledge_capacity", 100),
            "mid_term_similarity_threshold": kwargs.get("mid_term_similarity_threshold", 0.6),
            "mid_term_capacity": kwargs.get("mid_term_capacity", 200),
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
            start_time = time.time()
            # Enable log capture for this thread
            _thread_local.log_capture_list = []
            
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

            # Ingest newly completed history pairs (reference answers from Pipeline).
            # We manually call add_memory here so that MemoryOS stores the correct
            # reference content, NOT the model's generated output.
            # The auto add_memory inside get_response is disabled in memoryos.py.
            new_raw_inputs = []
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
                    new_raw_inputs.append(f"User: {current_user_msg}\nAssistant: {content}")
                    current_user_msg = None

                i += 1

            dataset_name = self.dataset_name
            # Update state cursor
            if dialog_id is not None:
                self._dialog_states[dialog_id]["last_ingested_idx"] = last_user_idx

            # Snapshot pre-generation state (read-only, no retrieve call to avoid
            # double-counting N_visit / heat in mid-term sessions).
            try:
                old_user_profile = client.user_long_term_memory.get_raw_user_profile(client.user_id)
                short_term_history = client.short_term_memory.get_all()
            except Exception as e:
                logger.warning(f"MemoryOS pre-generation info fetch failed: {e}")
                old_user_profile = "N/A"
                short_term_history = []

            # Monkey-patch retriever to capture results from the single retrieve
            # call inside get_response, avoiding a duplicate call that would
            # inflate N_visit / heat.
            _captured_retrieval = {}
            _orig_retrieve = client.retriever.retrieve_context
            def _patched_retrieve(*args, **kw):
                result = _orig_retrieve(*args, **kw)
                _captured_retrieval.update(result)
                return result
            client.retriever.retrieve_context = _patched_retrieve

            generate_start_time = time.time()
            response = client.get_response(query=final_query)
            generate_end_time = time.time()
            latency = generate_end_time - generate_start_time

            # Restore original method
            client.retriever.retrieve_context = _orig_retrieve

            retrieved_pages = _captured_retrieval.get("retrieved_pages", [])
            retrieved_user_knowledge = _captured_retrieval.get("retrieved_user_knowledge", [])

            # --- Diagnostic Logging ---
            if self.config.get('save_agent_logs', False):
                try:
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    log_dir = os.path.join(project_root, "output", "memoryos_logs", dataset_name)
                    os.makedirs(log_dir, exist_ok=True)
                
                    captured_logs = getattr(_thread_local, 'log_capture_list', [])
                    new_user_profile = client.user_long_term_memory.get_raw_user_profile(client.user_id)
                    profile_updated = (old_user_profile != new_user_profile)
                
                    # Format retrieved contexts
                    formatted_contexts = []
                    # Short term
                    for qa in short_term_history:
                        formatted_contexts.append({
                            "source": "memoryos_short_term",
                            "content": f"User: {qa.get('user_input', '')}\nAssistant: {qa.get('agent_response', '')}",
                            "score": None
                        })
                    # Mid term pages
                    for page in retrieved_pages:
                        formatted_contexts.append({
                            "source": "memoryos_mid_term",
                            "content": f"User: {page.get('user_input', '')}\nAssistant: {page.get('agent_response', '')}\nMeta info: {page.get('meta_info', 'N/A')}",
                            "score": None
                        })
                    # User Profile
                    formatted_contexts.append({
                        "source": "memoryos_user_profile",
                        "content": old_user_profile,
                        "score": 1.0
                    })
                    # Long term user knowledge
                    for k in retrieved_user_knowledge:
                        formatted_contexts.append({
                            "source": "memoryos_long_term_knowledge",
                            "content": k.get("knowledge"),
                            "score": None
                        })

                    diagnostic_data = {
                        "metadata": {
                            "dataset": dataset_name,
                            "dialog_id": dialog_id,
                            "turn_index": last_user_idx,
                            "query": final_query,
                            "timestamp": time.time(),
                            "latency_seconds": round(latency, 3)
                        },
                        "memory_update": {
                            "new_raw_inputs": new_raw_inputs,
                            "chunked_documents": [{"content": text, "meta_info": {}} for text in new_raw_inputs],
                            "extracted_knowledge": {
                                "memoryos_profile": new_user_profile if profile_updated else old_user_profile,
                                "memoryos_new_knowledge": [] # For brevity, we don't hook directly into the background update's returns
                            }
                        },
                        "retrieval": {
                            "search_queries": [final_query],
                            "retrieved_contexts": formatted_contexts
                        },
                        "generation": {
                            "generated_response": response
                        },
                        "system_logs": captured_logs
                    }
                
                    # Save into a per-dialog JSON file
                    dialog_file_name = f"dialog_{dialog_id}.json" if dialog_id is not None else "dialog_stateless.json"
                    dialog_file = os.path.join(log_dir, dialog_file_name)
                
                    if os.path.exists(dialog_file):
                        try:
                            with open(dialog_file, "r", encoding="utf-8") as f:
                                dialog_data = json.load(f)
                        except json.JSONDecodeError:
                            dialog_data = []
                    else:
                        dialog_data = []
                    
                    dialog_data.append(diagnostic_data)
                
                    with open(dialog_file, "w", encoding="utf-8") as f:
                        json.dump(dialog_data, f, ensure_ascii=False, indent=4)
                    
                except Exception as e:
                    logger.warning(f"MemoryOS diagnostic logging failed: {e}")
                finally:
                    # Cleanup thread-local log capture
                    _thread_local.log_capture_list = None
            # ---------------------------------------------

            return response if response else ""

        except Exception:
            logger.exception("MemoryOS generation error")
            raise