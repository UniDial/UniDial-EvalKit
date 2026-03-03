from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

from .base import BaseModel

logger = logging.getLogger(__name__)

try:
    from hipporag import HippoRAG
    from hipporag.utils.config_utils import BaseConfig
except ImportError:
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
        
        # 原始的模型名（用于日志 / 输出路径等，来自命令行 --model_name）
        self.raw_model_name = model_name
        # 实际调用底层 LLM 的模型名：
        # 支持像 "HippoRAG_deepseekchat" 这样的名字，只取下划线后面的部分作为真实 LLM 名
        if "_" in model_name:
            self.llm_model_name = model_name.split("_", 1)[1]
        else:
            self.llm_model_name = model_name

        if HippoRAG is None or BaseConfig is None:
            raise ImportError("HippoRAG package is missing. Install it or ensure it's in your Python path.")
        
        self.api_key = api_key 
        self.base_url = base_url 
        
        # Set environment variables for HippoRAG
        if self.api_key:
            os.environ["OPENAI_API_KEY"] = self.api_key
        if self.base_url:
            os.environ["OPENAI_BASE_URL"] = self.base_url
        
        # Default embedding model name,
        self.embedding_model_name = kwargs.get("embedding_model_name", "text-embedding-ada-002")
        
        # HippoRAG config parameters 
        self.linking_top_k = kwargs.get("linking_top_k", 10)
        self.retrieval_top_k = kwargs.get("retrieval_top_k", 200)
        self.synonymy_edge_sim_threshold = kwargs.get("synonymy_edge_sim_threshold", 0.8)
        self.qa_top_k = kwargs.get("qa_top_k", 5)

    def _create_hipporag_client(self, dialogue_texts: List[str]) -> HippoRAG:
        """
        Create and configure a HippoRAG client instance with indexed dialogue texts.
        This is similar to how OpenAI client is used in OpenAIModel.
        
        Args:
            dialogue_texts: List of dialogue text strings to index.
            
        Returns:
            Configured HippoRAG instance with indexed documents.
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
        Generate a response using HippoRAG.
        
        The method extracts dialogue history from messages (all messages except the last user message),
        indexes them into HippoRAG, and then uses the last user message as the query.
        
        Messages follow the unified format from schema:
        - {"role": "system"/"user"/"assistant", "content": str or None}
        - content is already processed as string in eval.py

        Args:
            messages: A list of message dictionaries with "role" and "content" keys.
                     Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
            temperature: Sampling temperature (not used by HippoRAG, kept for API compatibility).
            max_tokens: Maximum tokens to generate (not used by HippoRAG, kept for API compatibility).
            **kwargs: Additional generation parameters (not used, kept for API compatibility).

        Returns:
            The generated content string.
        """
        try:
            if not messages:
                return ""
            
            # Extract dialogue history and final query
            # All messages except the last user message are considered dialogue history
            dialogue_texts = []
            final_query = None
            
            # Find the last user message
            last_user_idx = -1
            for i in range(len(messages) - 1, -1, -1):
                role = messages[i].get("role", "").lower()
                if role == "user":
                    last_user_idx = i
                    break
            
            if last_user_idx == -1:
                # No user message found, return empty
                logger.warning("No user message found in messages")
                return ""
            
            # Extract dialogue history (all messages before the last user message)
            # Include all roles (system, user, assistant) as dialogue history
            for i in range(last_user_idx):
                msg = messages[i]
                content = msg.get("content")
                # content is already a string from schema processing, but handle None case
                if content:
                    dialogue_texts.append(str(content))
            
            # Extract final query (last user message)
            final_query = messages[last_user_idx].get("content")
            if not final_query:
                logger.warning("Final query is empty")
                return ""
            final_query = str(final_query)

            # Special handling for mt_eval: the very first user turn (turn_id=0)
            # often contains a long context + 'Instruction: ...'.
            # For the first generation (turn 1), we want:
            # - context part -> indexed into HippoRAG as document
            # - instruction part (after 'Instruction:') -> used as the query
            if last_user_idx == 0 and "Instruction:" in final_query:
                try:
                    context_part, instr_part = final_query.split("Instruction:", 1)
                    context_part = context_part.strip()
                    instr_part = instr_part.strip()

                    # Use long context as a document
                    if context_part:
                        dialogue_texts.append(context_part)

                    # Use only the instruction text as query if available
                    if instr_part:
                        final_query = instr_part
                except Exception as split_err:
                    logger.warning(f"Failed to split initial user turn on 'Instruction:': {split_err}")
            
            # Create HippoRAG client and index dialogue history
            hipporag = self._create_hipporag_client(dialogue_texts)
            
            # Generate answer using HippoRAG (same as OpenAI client usage pattern)
            queries_solutions, response_messages, metadata = hipporag.rag_qa(queries=[final_query])
            
            if queries_solutions and len(queries_solutions) > 0:
                answer = queries_solutions[0].answer
                return answer if answer else ""
            
            return ""
            
        except Exception as e:
            logger.error(f"HippoRAG generation error: {e}")
            raise e
