from __future__ import annotations

import logging
import os
import sys
import tempfile
import threading
from typing import Any, Dict, List, Optional

from .base import BaseModel

logger = logging.getLogger(__name__)

try:
    from src.hipporag import HippoRAG
    from src.hipporag.utils.config_utils import BaseConfig
    logger.info("Successfully imported HippoRAG from src.hipporag")
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to import HippoRAG from src.hipporag: {e}")
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
        self.linking_top_k = kwargs.get("linking_top_k", 10)
        self.retrieval_top_k = kwargs.get("retrieval_top_k", 10)
        self.synonymy_edge_sim_threshold = kwargs.get("synonymy_edge_sim_threshold", 0.8)
        self.qa_top_k = kwargs.get("qa_top_k", 5)
        
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
            if last_user_idx == 0:
                ingest_end_idx = 1
            
            # Ingest new history
            for i in range(start_idx, ingest_end_idx):
                if i >= len(messages): break # Safety check
                content = messages[i].get("content")
                if content:
                    new_texts.append(str(content))
            
            # Update state index
            state["last_ingested_idx"] = ingest_end_idx
            
            # Initialize or update HippoRAG
            if state["hipporag"] is None:
                # First time initialization
                logger.info(f"Initializing HippoRAG for dialog {dialog_id} with {len(new_texts)} documents")
                
                # Aggregate any accumulated texts
                state["dialogue_texts"].extend(new_texts)
                
                # Create client (indexing happens inside)
                if state["dialogue_texts"]:
                     state["hipporag"] = self._create_hipporag_client(state["dialogue_texts"])
            else:
                # Update existing index with new texts
                if new_texts:
                    logger.info(f"Incrementally indexing {len(new_texts)} new documents for dialog {dialog_id}")
                    state["dialogue_texts"].extend(new_texts)
                    try:
                        state["hipporag"].index(docs=new_texts) 
                    except Exception as e:
                        logger.error(f"Error during incremental indexing: {e}. Falling back to full re-index.")
                        # Fallback: Re-create client with full history if incremental fails
                        state["hipporag"] = self._create_hipporag_client(state["dialogue_texts"]) 
            
            hipporag = state["hipporag"]
            
            # Generate answer
            if hipporag:
                queries_solutions, _, _ = hipporag.rag_qa(queries=[final_query])
                
                if queries_solutions and len(queries_solutions) > 0:
                    answer = queries_solutions[0].answer
                    return answer if answer else ""
            
            return ""
            
        except Exception as e:
            logger.error(f"HippoRAG generation error: {e}")
            raise e
