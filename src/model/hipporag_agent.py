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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _ensure_hipporag_import_paths() -> None:
    """Make bundled HippoRAG importable for its internal absolute imports."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    hipporag_root = os.path.join(current_dir, "HippoRAG")
    hipporag_src = os.path.join(hipporag_root, "src")

    for path in (hipporag_root, hipporag_src):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


_ensure_hipporag_import_paths()

try:
    from .HippoRAG.src.hipporag import HippoRAG
    from .HippoRAG.src.hipporag.utils.config_utils import BaseConfig
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
        
        if self.config["save_agent_logs"]:
            self.logs_output_dir = self.config["agent_logs_output_dir"]
            os.makedirs(self.logs_output_dir, exist_ok=True)

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
        **kwargs: Any
    ) -> str:
        """
        Generate a response using HippoRAG with stateful dialog management.
        """
        try:
            start_time = time.time()
            
            if not messages:
                return ""
            
            # extract the final user query
            final_query = "" 
            last_msg = messages[-1]
            if last_msg.get("role", "").lower() == "user":
                final_query = str(last_msg.get("content", ""))
     
            # if no final query, return empty string
            if not final_query:
                return ""
            
            # logger.info(f"Final Query sent to HippoRAG: {final_query}")
            
            # Identify dialog state
            dialog_id = kwargs.get("dialog_id")
            if dialog_id is not None:
                with self._state_lock:
                    state = self._dialog_states.get(dialog_id)

            else:
                 # Stateless mode fallback
                 state = {
                    "hipporag": None,
                    "last_ingested_idx": 0,
                    "dialogue_texts": []
                }
            
            
            start_idx = state["last_ingested_idx"]
            last_user_idx = len(messages) - 1
            
            # Identify new history to ingest
            new_texts = []
            
            # Validation
            if start_idx < 0 or start_idx > last_user_idx:
                start_idx = 0

            # Ingest new history
            current_user_msg = None
            for i in range(start_idx, last_user_idx):
                if i >= len(messages): break # Safety check
                msg = messages[i]
                role = str(msg.get("role", "")).lower()
                content = str(msg.get("content", "")).strip()
                
                if role == "user":
                    if current_user_msg is not None:
                        new_texts.append(f"User: {current_user_msg}")
                    current_user_msg = content
                elif role == "assistant":
                    if current_user_msg is not None:
                        new_texts.append(f"User: {current_user_msg}\nAssistant: {content}")
                        current_user_msg = None
                    else:
                        new_texts.append(f"Assistant: {content}")
            
            
            if current_user_msg is not None:
                new_texts.append(f"User: {current_user_msg}")
            
            # Update state index
            state["last_ingested_idx"] = last_user_idx
            
            # Initialize or update HippoRAG
            if state["hipporag"] is None:
                # First time initialization
                logger.info(f"Initializing HippoRAG for dialog {dialog_id} with {len(new_texts)} documents")
                # logger.info(f"New documents to index: {new_texts}")
                
                # Aggregate any accumulated texts
                state["dialogue_texts"].extend(new_texts)
                
                # Create client (indexing happens inside if texts exist)
                state["hipporag"] = self._create_hipporag_client(state["dialogue_texts"])
            else:
                # Update existing index with new texts
                if new_texts:
                    logger.info(f"Incrementally indexing {len(new_texts)} new documents for dialog {dialog_id}")
                    # logger.info(f"New documents to index: {new_texts}")
                    state["dialogue_texts"].extend(new_texts)
                    try:
                        state["hipporag"].index(docs=new_texts)
                        state["hipporag"].ready_to_retrieve = False
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
                    # if new_extracted_triples:
                    #     logger.info(f"Extracted triples for new documents: {json.dumps(new_extracted_triples, ensure_ascii=False)}")
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
                    # try:
                    result = hipporag.llm_model.infer([{"role": "user", "content": final_query}])
                    if isinstance(result, tuple):
                        response = result[0]
                    else:
                        response = str(result)
                    
                else:
                    queries_solutions, all_response_messages, all_metadata = hipporag.rag_qa(queries=[final_query])
                    
                    if queries_solutions and len(queries_solutions) > 0:
                        retrieved_docs = queries_solutions[0].docs
                        # logger.info(f"Retrieved top-K documents/contexts:\n{retrieved_docs}")

                    if all_response_messages and len(all_response_messages) > 0:
                        # logger.info(f"Final Generated Response: {all_response_messages[0]}")
                        response = all_response_messages[0]

            # --- Diagnostic Logging ---
            if self.config.get('save_agent_logs', False):
                # try:
                    
                    # captured_logs = getattr(_thread_local, 'log_capture_list', [])
                
                    end_time = time.time()
                    latency = end_time - start_time
                

                    diagnostic_data = {
                        "metadata": {
                            "dialog_id": dialog_id,
                            "turn_index": last_user_idx + 1, # assistant turn index is the next turn index of the user turn
                            "query": final_query,
                            "timestamp": time.time(),
                            "latency_seconds": round(latency, 3),
                            "all_metadata": all_metadata[0]
                        },
                        "memory_update": {
                            "chunked_documents": new_texts,
                            "extracted_knowledge": {
                                "hipporag_triples": new_extracted_triples
                            } if new_extracted_triples else {}
                        },
                        "retrieval": {
                            "search_queries": [final_query],
                            "retrieved_contexts": retrieved_docs
                        },
                        "generation": {
                            "generated_response": response
                        },
                        # "system_logs": captured_logs
                    }
                
                    # Save into a per-dialog JSON file
                    dialog_file_name = f"dialog_{dialog_id}.json" if dialog_id is not None else "dialog_stateless.json"
                    dialog_file = os.path.join(self.logs_output_dir, dialog_file_name)
                
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
                    
               
            return response
            
        except Exception as e:
            logger.exception("HippoRAG generation error")
            raise e
