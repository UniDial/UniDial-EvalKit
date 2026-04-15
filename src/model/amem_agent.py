import os
import sys
import logging
import json
import time
import threading
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
logging.getLogger().addHandler(_capture_handler)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

    def answer_question(self, question: str) -> tuple[str, str, Any]:
        try:
            keywords = self.generate_query_llm(question)
            logger.info(f"Generated keywords for retrieval: {keywords}")
            
            raw_context = self.retrieve_memory(keywords, k=self.retrieve_k)
            logger.info(f"Retrieved memories:\n{raw_context}")
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
            logger.info(f"Final Generated Response: {response}")
            return response, keywords, raw_context
        except Exception as e:
            logger.error(f"Error in answer_question: {e}")
            return "", "", ""

class AMemModel(BaseModel):
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        retrieve_k: int = 3,
        **kwargs: Any
    ) -> None:
        super().__init__(model_name, **kwargs)
        
        self.raw_model_name = model_name
        self.llm_backend = "openai"
        self.llm_model = model_name
        
        if AgenticMemorySystem is None:
            raise ImportError("AgenticMemory modules not found. ")
            
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
            start_time = time.time()
            # Enable log capture for this thread
            _thread_local.log_capture_list = []
            
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
            start_idx = int(state.get("last_ingested_idx", 0))
            if start_idx < 0 or start_idx > last_user_idx:
                start_idx = 0

            current_user_msg = None
            new_texts = []
            extracted_notes = []
            for i in range(start_idx, last_user_idx):
                msg = messages[i]
                role = str(msg.get("role", "")).lower()
                content = str(msg.get("content", "")).strip()
                if not content:
                    continue
                    
                if role == "user":
                    if current_user_msg is not None:
                        new_text = f"User input: {current_user_msg}"
                        agent.add_memory(content=new_text)
                        new_texts.append(new_text)
                    current_user_msg = content
                elif role == "assistant":
                    if current_user_msg is not None:
                        new_text = f"User input: {current_user_msg}\nAI response: {content}"
                        agent.add_memory(content=new_text)
                        new_texts.append(new_text)
                        current_user_msg = None
                    else:
                        new_text = f"AI response: {content}"
                        agent.add_memory(content=new_text)
                        new_texts.append(new_text)

            if current_user_msg is not None:
                new_text = f"User input: {current_user_msg}"
                agent.add_memory(content=new_text)
                new_texts.append(new_text)

            # Capture extracted knowledge from newly added memories
            if new_texts:
                logger.info(f"New memories to index:\n" + "\n".join(new_texts))
                try:
                    for note in agent.memory_system.memories.values():
                        if note.content in new_texts:
                            extracted_notes.append({
                                "content": note.content,
                                "context": note.context,
                                "keywords": note.keywords,
                                "tags": getattr(note, 'tags', [])
                            })
                except Exception as e:
                    logger.warning(f"Failed to extract knowledge from AMem notes: {e}")

            state["last_ingested_idx"] = last_user_idx

            logger.info(f"Final Query sent to AMem: {final_query}")

            keywords = ""
            raw_context = ""
            # 如果没有任何历史记忆被保存（比如第一轮），直接回答
            if not state["agent"].memory_system.memories:
                logger.info("No memories available. Direct LLM generation.")
                prediction = agent.memory_system.llm_controller.llm.get_completion(
                    final_query,
                    temperature=temperature
                )
                logger.info(f"Final Generated Response: {prediction}")
            else:
                prediction, keywords, raw_context = agent.answer_question(final_query)
            
            try:
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
                
            # --- Diagnostic Logging ---
            if self.config.get('save_agent_logs', False):
                try:
                    dataset_name = self.dataset_name
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    log_dir = os.path.join(project_root, "output", "amem_logs", dataset_name)
                    os.makedirs(log_dir, exist_ok=True)
                
                    captured_logs = getattr(_thread_local, 'log_capture_list', [])
                
                    end_time = time.time()
                    latency = end_time - start_time
                
                    retrieved_contexts = []
                    if raw_context:
                        retrieved_contexts.append({
                            "source": "amem_vector_db",
                            "content": str(raw_context),
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
                            "new_raw_inputs": new_texts,
                            "chunked_documents": [{"content": text, "meta_info": {}} for text in new_texts],
                            "extracted_knowledge": {
                                "amem_notes": extracted_notes
                            } if extracted_notes else {}
                        },
                        "retrieval": {
                            "search_queries": [keywords] if keywords else [],
                            "retrieved_contexts": retrieved_contexts
                        },
                        "generation": {
                            "generated_response": str(prediction).strip()
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
                    logger.warning(f"AMem diagnostic logging failed: {e}")
                finally:
                    # Cleanup thread-local log capture
                    _thread_local.log_capture_list = None
            # ---------------------------------------------
            
            return str(prediction).strip()
            
        except Exception as e:
            # Ensure cleanup on error
            if hasattr(_thread_local, 'log_capture_list'):
                _thread_local.log_capture_list = None
            logger.error(f"AMem generation error: {e}")
            raise e

