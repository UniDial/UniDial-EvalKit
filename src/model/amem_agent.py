import os
import sys
import logging
import json
import ast
import re
import time
import threading
from typing import Any, Dict, List, Optional

from .base import BaseModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Try importing the components
try:
    from .A_Mem.memory_layer import AgenticMemorySystem, LLMController
except ImportError as e:
    logger.error(f"Failed to import AgenticMemory components: {e}")
    import traceback
    traceback.print_exc()
    AgenticMemorySystem = None
    LLMController = None

class AdvancedMemAgent:
    """Advanced memory agent that follows test_advanced.py approach"""
    def __init__(self, embedding_model_name, model, backend, api_key, api_base, retrieve_k=10, temperature= 0.7):
        self.memory_system = AgenticMemorySystem(
            model_name=embedding_model_name,
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
        # print(backend, model, api_key, api_base)
        # exit(0)

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
            response = self.retriever_llm.llm.get_completion(prompt, response_format={"type": "json_schema", "json_schema": {
                            "name": "response",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "keywords": {
                                        "type": "string",
                                    }
                                },
                                "required": ["keywords"],
                                "additionalProperties": False
                            },
                            "strict": True
                        }}, temperature=self.temperature)
            try:
                response = json.loads(response)["keywords"]
            except:
                response = response.strip()
            return response
        except Exception as e:
            logger.error(f"Error in generate_query_llm: {e}")
            raise e

    def answer_question(self, question: str) -> tuple[str, str, Any]:
        try:
            keywords = self.generate_query_llm(question)
            # logger.info(f"Generated keywords for retrieval: {keywords}")
            
            raw_context = self.retrieve_memory(keywords, k=self.retrieve_k)
            # logger.info(f"Retrieved memories:\n{raw_context}")
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
                response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "answer": {
                                    "type": "string",
                                }
                            },
                            "required": ["answer"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }},
                temperature=self.temperature
            )
            # logger.info(f"Final Generated Response: {response}")
            return response, keywords, raw_context
        except Exception as e:
            logger.error(f"Error in answer_question: {e}")
            raise e

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
        
        self.llm_backend = "openai"
        self.llm_model = model_name
        self.embedding_model_name = self.config["embedding_model_name"]

        
        if AgenticMemorySystem is None:
            raise ImportError("AgenticMemory modules not found. ")
            
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "")
        self.retrieve_k = retrieve_k
        # dialog_id -> {"agent": AdvancedMemAgent, "last_ingested_idx": int}
        self._dialog_states: Dict[int, Dict[str, Any]] = {}
        self._state_lock = threading.Lock()
        
        if self.config["save_agent_logs"]:
            self.logs_output_dir = self.config["agent_logs_output_dir"]
            os.makedirs(self.logs_output_dir, exist_ok=True)

    def _create_agent(self, temperature: float) -> AdvancedMemAgent:
        return AdvancedMemAgent(
            model=self.llm_model,
            backend=self.llm_backend,
            api_key=self.api_key,
            api_base=self.base_url,
            retrieve_k=self.retrieve_k,
            temperature=temperature,
            embedding_model_name=self.embedding_model_name,
        )

    def begin_dialog(self, dialog_id: Optional[int] = None) -> None:
        if dialog_id is None:
            return
        # Per-dialog state is created lazily in generate() when dialog_id is set.

    def end_dialog(self, dialog_id: Optional[int] = None) -> None:
        if dialog_id is None:
            return
        with self._state_lock:
            self._dialog_states.pop(dialog_id, None)

    @staticmethod
    def _parse_list_field(raw_value: str) -> List[Any]:
        text = (raw_value or "").strip()
        if not text:
            return []
        try:
            value = ast.literal_eval(text)
            if isinstance(value, list):
                return value
            return [value]
        except Exception:
            return [text]

    @classmethod
    def _parse_retrieved_contexts(cls, raw_context: Any) -> List[Dict[str, Any]]:
        if not raw_context:
            return []
        if isinstance(raw_context, list):
            return raw_context
        if not isinstance(raw_context, str):
            return [{"source": "amem_vector_db", "content": str(raw_context), "score": None}]

        pattern = (
            r"talk start time:(?P<talk_start_time>.*?)"
            r"memory content:\s*(?P<memory_content>.*?)"
            r"memory context:\s*(?P<memory_context>.*?)"
            r"memory keywords:\s*(?P<memory_keywords>.*?)"
            r"memory tags:\s*(?P<memory_tags>.*?)(?=talk start time:|$)"
        )
        parsed_items: List[Dict[str, Any]] = []
        for match in re.finditer(pattern, raw_context, re.DOTALL):
            parsed_items.append({
                "talk_start_time": match.group("talk_start_time").strip(),
                "memory_content": match.group("memory_content").strip(),
                "memory_context": match.group("memory_context").strip(),
                "memory_keywords": cls._parse_list_field(match.group("memory_keywords")),
                "memory_tags": cls._parse_list_field(match.group("memory_tags")),
                "score": None,
            })

        if parsed_items:
            return parsed_items
        return [{"source": "amem_vector_db", "content": raw_context, "score": None}]

    def generate(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        # max_tokens: int = 1024,
        **kwargs: Any
    ) -> str:
        try:
            start_time = time.time()
            # Enable log capture for this thread
            # _thread_local.log_capture_list = []
            
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
                # logger.info(f"New memories to index:\n" + "\n".join(new_texts))
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

            # logger.info(f"Final Query sent to AMem: {final_query}")

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
                # dataset_name = self.dataset_name
                        
                end_time = time.time()
                latency = end_time - start_time
            
                retrieved_contexts = self._parse_retrieved_contexts(raw_context)

                diagnostic_data = {
                    "metadata": {
                        # "dataset": dataset_name,
                        "dialog_id": dialog_id,
                        "turn_index": last_user_idx + 1, # assistant turn index is the next turn index of the user turn
                        "query": final_query,
                        "timestamp": time.time(),
                        "latency_seconds": round(latency, 3)
                    },
                    "memory_update": {
                        "chunked_documents": new_texts,
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
                    # "system_logs": captured_logs
                }
            
                # Save into a per-dialog JSON file
                dialog_file_name = f"dialog_{dialog_id}.json"
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
            
            return str(prediction).strip()

        except Exception as e:
            logger.exception("AMem generation error")
            raise e
