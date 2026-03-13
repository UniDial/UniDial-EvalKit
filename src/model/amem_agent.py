import os
import sys
import logging
import json
import threading
from typing import Any, Dict, List, Optional

from .base import BaseModel

logger = logging.getLogger(__name__)

# Try importing the components
try:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    from AgenticMemory.memory_layer import AgenticMemorySystem, LLMController
except ImportError as e:
    logger.error(f"Failed to import AgenticMemory components: {e}")
    import traceback
    traceback.print_exc()
    AgenticMemorySystem = None
    LLMController = None

class AdvancedMemAgent:
    """Advanced memory agent that follows test_advanced.py approach"""
    def __init__(self, model, backend, api_key, api_base, retrieve_k=10, temperature=0.7):
        self.memory_system = AgenticMemorySystem(
            model_name='all-MiniLM-L6-v2',
            llm_backend=backend,
            llm_model=model,
            api_key=api_key,
            api_base=api_base
        )
        self.retriever_llm = LLMController(
            backend=backend,
            model=model,
            api_key=api_key,
            api_base=api_base
        )
        self.retrieve_k = retrieve_k
        self.temperature = temperature

    def add_memory(self, content, time=None):
        self.memory_system.add_note(content, time=time)

    def retrieve_memory(self, content, k=10):
        return self.memory_system.find_related_memories_raw(content, k=k)

    def generate_query_llm(self, question):
        prompt = f"""Given the following question, generate several keywords separated by commas.

                Question: {question}

                Example response:
                keyword1, keyword2, keyword3"""
            
        try:
            response = self.retriever_llm.llm.get_completion(prompt, temperature=0.1)
            return response.strip()
        except Exception as e:
            logger.error(f"Error in generate_query_llm: {e}")
            return question.strip()

    def answer_question(self, question: str) -> str:
        try:
            keywords = self.generate_query_llm(question)
            raw_context = self.retrieve_memory(keywords, k=self.retrieve_k)
            memories_str = raw_context
            
            user_prompt = f""" 
             Based on following dialogue history and related memories, please answer final question. Ensure your answer is based on the content discussed in the dialogues. 
             User Memories: 
             {memories_str} 
             Final Question
             {question}
             """.strip()
            
            response = self.memory_system.llm_controller.llm.get_completion(
                user_prompt,
                temperature=self.temperature
            )
            return response
        except Exception as e:
            logger.error(f"Error in answer_question: {e}")
            return ""

class AMemModel(BaseModel):
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        retrieve_k: int = 10,
        **kwargs: Any
    ) -> None:
        super().__init__(model_name, **kwargs)
        
        self.raw_model_name = model_name
        self.llm_backend = "openai"
        self.llm_model = model_name
        
        if AgenticMemorySystem is None:
            raise ImportError("AgenticMemory modules not found. Ensure AgenticMemory/A-mem-main is available.")
            
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "")
        self.retrieve_k = retrieve_k
        # dialog_id -> {"agent": AdvancedMemAgent, "last_ingested_idx": int}
        self._dialog_states: Dict[int, Dict[str, Any]] = {}
        self._state_lock = threading.Lock()

    def _create_agent(self, temperature: float) -> AdvancedMemAgent:
        return AdvancedMemAgent(
            model=self.llm_model,
            backend=self.llm_backend,
            api_key=self.api_key,
            api_base=self.base_url,
            retrieve_k=self.retrieve_k,
            temperature=temperature,
        )

    def begin_dialog(self, dialog_id: Optional[int] = None, **kwargs: Any) -> None:
        if dialog_id is None:
            return
        temperature = float(kwargs.get("temperature", 0.7))
        with self._state_lock:
            if dialog_id not in self._dialog_states:
                self._dialog_states[dialog_id] = {
                    "agent": self._create_agent(temperature=temperature),
                    "last_ingested_idx": 0,
                }

    def end_dialog(self, dialog_id: Optional[int] = None, **kwargs: Any) -> None:
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
        try:
            if not messages:
                return ""
                
            final_query = ""
            for msg in reversed(messages):
                if msg.get("role", "").lower() == "user":
                    final_query = str(msg.get("content", ""))
                    break
                    
            if not final_query:
                return ""
            dialog_id = kwargs.get("dialog_id")
            if dialog_id is not None:
                with self._state_lock:
                    state = self._dialog_states.get(dialog_id)
                    if state is None:
                        state = {
                            "agent": self._create_agent(temperature=temperature),
                            "last_ingested_idx": 0,
                        }
                        self._dialog_states[dialog_id] = state
            else:
                # Keep backward compatibility for callers that do not pass dialog_id:
                # use an ephemeral per-call agent/state.
                state = {
                    "agent": self._create_agent(temperature=temperature),
                    "last_ingested_idx": 0,
                }

            agent = state["agent"]
            agent.temperature = temperature

            # By dataset/pipeline contract, the last input message is always the current user query.
            last_user_idx = len(messages) - 1

            # Only ingest newly added history before current final user query.
            # Current generation output is intentionally NOT ingested here because
            # the caller may choose generated/reference history outside this model.
            start_idx = int(state.get("last_ingested_idx", 0))
            if start_idx < 0 or start_idx > last_user_idx:
                start_idx = 0

            for i in range(start_idx, last_user_idx):
                msg = messages[i]
                role = str(msg.get("role", "")).lower()
                content = str(msg.get("content", ""))
                if not content:
                    continue
                if role == "user":
                    agent.add_memory(content=f"User input: {content}")
                elif role == "assistant":
                    agent.add_memory(content=f"AI response: {content}")

            state["last_ingested_idx"] = last_user_idx

            prediction = agent.answer_question(final_query)
            
            try:
                # Some implementations return a JSON with "answer" key
                if "{" in prediction and "}" in prediction:
                    import re
                    # Try to extract pure JSON
                    json_match = re.search(r'\{.*\}', prediction, re.DOTALL)
                    if json_match:
                        parsed = json.loads(json_match.group(0))
                        if "answer" in parsed:
                            prediction = parsed["answer"]
            except Exception:
                pass
                
            return str(prediction).strip()
            
        except Exception as e:
            logger.error(f"AMem generation error: {e}")
            raise e

