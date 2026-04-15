from __future__ import annotations

import logging
import os
import sys
import tempfile
import threading
import json
import time
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
    from hipporag import HippoRAG
    from hipporag.utils.config_utils import BaseConfig
    logger.info("Successfully imported HippoRAG")
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to import HippoRAG: {e}")
    HippoRAG = None
    BaseConfig = None


class HippoRAGModel(BaseModel):
    """
    HippoRAG-v2 Model Wrapper for chat completion.
    Uses HippoRAG to index dialogue history and generate responses.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize the HippoRAG model.
        
        Args:
            model_name: The name of the LLM model to use (e.g., "deepseek-chat").
            api_key: API key for the LLM (optional if env var set).
            base_url: Custom API base URL (optional).
            **kwargs: Additional configuration parameters.
        """
        super().__init__(model_name, **kwargs)
        self.raw_model_name = model_name
        if "_" in model_name:
            self.llm_model_name = model_name.split("_", 1)[1]
        else:
            self.llm_model_name = model_name

        if HippoRAG is None or BaseConfig is None:
            raise ImportError("HippoRAG package is missing. Install it or ensure it's in your Python path.")
        
        self.api_key = api_key 
        self.base_url = base_url 
        
        # HippoRAG specific configuration
        self.embedding_model_name = kwargs.get("embedding_model_name", "Transformers/sentence-transformers/all-MiniLM-L6-v2")
        self.linking_top_k = kwargs.get("linking_top_k", 5)
        self.retrieval_top_k = kwargs.get("retrieval_top_k", 10)
        self.synonymy_edge_sim_threshold = kwargs.get("synonymy_edge_sim_threshold", 0.8)
        self.qa_top_k = kwargs.get("qa_top_k", 10)
        
        # Set environment variables for HippoRAG
        if self.api_key:
            os.environ["OPENAI_API_KEY"] = self.api_key
        if self.base_url:
            os.environ["OPENAI_BASE_URL"] = self.base_url
            
        # Maintain dialog_id to HippoRAG instance mapping
        self._dialog_states: Dict[int, Dict[str, Any]] = {}
        self._state_lock = threading.Lock()

    def begin_dialog(self, dialog_id: Optional[int] = None, **kwargs: Any) -> None:
        """
        Allocate state for a new dialog.
        """
        if dialog_id is None:
            return
        
        with self._state_lock:
            if dialog_id not in self._dialog_states:
                self._dialog_states[dialog_id] = {
                    "hipporag": None,
                    "last_ingested_idx": 0,
                    "dialogue_texts": []
                }

    def end_dialog(self, dialog_id: Optional[int] = None, **kwargs: Any) -> None:
        """
        Cleanup resources for a finished dialog.
        """
        if dialog_id is None:
            return
        
        with self._state_lock:
            if dialog_id in self._dialog_states:
                self._dialog_states.pop(dialog_id, None)

    def _create_hipporag_client(self, dialogue_texts: List[str]) -> HippoRAG:
        """
        Create and configure a HippoRAG client instance with indexed dialogue texts.
        
        """
        # Use temporary directory for HippoRAG data 
        temp_dir = tempfile.mkdtemp(prefix="hipporag_")
        
        # Create custom config
        config = BaseConfig(
            linking_top_k=self.linking_top_k,
            retrieval_top_k=self.retrieval_top_k,
            synonymy_edge_sim_threshold=self.synonymy_edge_sim_threshold,
            qa_top_k=self.qa_top_k,
        )
        
        # Initialize HippoRAG client
        hipporag = HippoRAG(
            global_config=config,
            save_dir=temp_dir,
            llm_model_name=self.llm_model_name,
            llm_base_url=self.base_url,
            embedding_model_name=self.embedding_model_name
        )
        
        # Index the dialogue texts
        if dialogue_texts:
            hipporag.index(docs=dialogue_texts)
        
        return hipporag

    def generate(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any
    ) -> str:
        """
        Generate a response using HippoRAG with stateful dialog management.
        """
        try:
            start_time = time.time()
            # Enable log capture for this thread
            _thread_local.log_capture_list = []
            
            if not messages:
                return ""
            
            # Find the last user message (the query)
            last_user_idx = -1
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role", "").lower() == "user":
                    last_user_idx = i
                    break
            
            if last_user_idx == -1:
                logger.warning("No user message found in messages")
                return ""
            
            final_query = str(messages[last_user_idx].get("content", ""))
            logger.info(f"Final Query sent to HippoRAG: {final_query}")
            
            # Identify dialog state
            dialog_id = kwargs.get("dialog_id")
            if dialog_id is not None:
                with self._state_lock:
                    state = self._dialog_states.get(dialog_id)
                    if state is None:
                        # Auto-initialize if begin_dialog missed (though pipeline should call it)
                        state = {
                            "hipporag": None,
                            "last_ingested_idx": 0,
                            "dialogue_texts": []
                        }
                        self._dialog_states[dialog_id] = state
            else:
                 # Stateless mode fallback
                 state = {
                    "hipporag": None,
                    "last_ingested_idx": 0,
                    "dialogue_texts": []
                }
            
            
            start_idx = state["last_ingested_idx"]
            
            # Identify new history to ingest
            new_texts = []
            
            # Validation
            if start_idx < 0 or start_idx > last_user_idx:
                start_idx = 0

            # Determine ingestion range
            ingest_end_idx = last_user_idx
            
            # Ingest new history
            current_user_msg = None
            for i in range(start_idx, ingest_end_idx):
                if i >= len(messages): break # Safety check
                msg = messages[i]
                role = str(msg.get("role", "")).lower()
                content = str(msg.get("content", "")).strip()
                
                if not content:
                    continue
                    
                if role == "user":
                    if current_user_msg is not None:
                        new_texts.append(f"User: {current_user_msg}")
                    current_user_msg = content
                elif role == "assistant":
                    if current_user_msg is not None:
                        # 组合成完整的 QA 块
                        new_texts.append(f"User: {current_user_msg}\nAssistant: {content}")
                        current_user_msg = None
                    else:
                        new_texts.append(f"Assistant: {content}")
            
            # 如果最后剩下一个孤立的 user
            if current_user_msg is not None:
                new_texts.append(f"User: {current_user_msg}")
            
            # Update state index
            state["last_ingested_idx"] = ingest_end_idx
            
            # Initialize or update HippoRAG
            if state["hipporag"] is None:
                # First time initialization
                logger.info(f"Initializing HippoRAG for dialog {dialog_id} with {len(new_texts)} documents")
                logger.info(f"New documents to index: {new_texts}")
                
                # Aggregate any accumulated texts
                state["dialogue_texts"].extend(new_texts)
                
                # Create client (indexing happens inside if texts exist)
                state["hipporag"] = self._create_hipporag_client(state["dialogue_texts"])
            else:
                # Update existing index with new texts
                if new_texts:
                    logger.info(f"Incrementally indexing {len(new_texts)} new documents for dialog {dialog_id}")
                    logger.info(f"New documents to index: {new_texts}")
                    state["dialogue_texts"].extend(new_texts)
                    try:
                        state["hipporag"].index(docs=new_texts) 
                    except Exception as e:
                        logger.error(f"Error during incremental indexing: {e}. Falling back to full re-index.")
                        # Fallback: Re-create client with full history if incremental fails
                        state["hipporag"] = self._create_hipporag_client(state["dialogue_texts"]) 
            
            hipporag = state["hipporag"]
            
            # Fetch newly extracted triples for logging
            new_extracted_triples = []
            if hipporag and new_texts:
                try:
                    all_openie_info, _ = hipporag.load_existing_openie([])
                    for info in all_openie_info:
                        if info.get('passage') in new_texts:
                            new_extracted_triples.append({
                                "passage": info.get('passage'),
                                "entities": info.get('extracted_entities', []),
                                "triples": info.get('extracted_triples', [])
                            })
                    if new_extracted_triples:
                        logger.info(f"Extracted triples for new documents: {json.dumps(new_extracted_triples, ensure_ascii=False)}")
                except Exception as e:
                    logger.warning(f"Failed to fetch extracted triples from HippoRAG: {e}")
            
            # Generate answer
            response = ""
            retrieved_docs = []
            if hipporag:
                # Bypass RAG completely if no documents were ever indexed
                if not state["dialogue_texts"]:
                    # Direct generation bypassing HippoRAG's retrieve/index steps
                    logger.info("No documents indexed. Falling back to direct LLM generation.")
                    try:
                        result = hipporag.llm_model.infer([{"role": "user", "content": final_query}])
                        if isinstance(result, tuple):
                            response = result[0]
                        else:
                            response = str(result)
                    except AttributeError:
                        logger.warning("hipporag.llm_model.infer failed, falling back to openai_client direct call.")
                        raw_resp = hipporag.llm_model.openai_client.chat.completions.create(
                            model=hipporag.llm_model.global_config.llm_name,
                            messages=[{"role": "user", "content": final_query}],
                            max_tokens=1024,
                            temperature=0.7,
                        )
                        response = raw_resp.choices[0].message.content
                else:
                    queries_solutions, all_response_messages, _ = hipporag.rag_qa(queries=[final_query])
                    
                    if queries_solutions and len(queries_solutions) > 0:
                        retrieved_docs = queries_solutions[0].docs
                        logger.info(f"Retrieved top-K documents/contexts:\n{retrieved_docs}")

                    if all_response_messages and len(all_response_messages) > 0:
                        logger.info(f"Final Generated Response: {all_response_messages[0]}")
                        response = all_response_messages[0]

            # --- Diagnostic Logging ---
            if self.config.get('save_agent_logs', False):
                try:
                    dataset_name = self.dataset_name
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    log_dir = os.path.join(project_root, "output", "hipporag_logs", dataset_name)
                    os.makedirs(log_dir, exist_ok=True)
                
                    captured_logs = getattr(_thread_local, 'log_capture_list', [])
                
                    end_time = time.time()
                    latency = end_time - start_time
                
                    # Format retrieved contexts for HippoRAG
                    formatted_contexts = []
                    for doc in retrieved_docs:
                        formatted_contexts.append({
                            "source": "hipporag_graph",
                            "content": str(doc),
                            "score": None # If PPR score is available we could put it here, otherwise None
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
                            "new_raw_inputs": new_texts,
                            "chunked_documents": [{"content": text, "meta_info": {}} for text in new_texts],
                            "extracted_knowledge": {
                                "hipporag_triples": new_extracted_triples
                            } if new_extracted_triples else {}
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
                    logger.warning(f"HippoRAG diagnostic logging failed: {e}")
                finally:
                    # Cleanup thread-local log capture
                    _thread_local.log_capture_list = None
            # ---------------------------------------------
            
            return response
            
        except Exception as e:
            # Ensure cleanup on error
            if hasattr(_thread_local, 'log_capture_list'):
                _thread_local.log_capture_list = None
            logger.error(f"HippoRAG generation error: {repr(e)}", exc_info=True)
            raise e
