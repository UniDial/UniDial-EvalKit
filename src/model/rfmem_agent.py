from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI


from .base import BaseModel

logger = logging.getLogger(__name__)

def _ensure_rfmem_import_paths() -> None:
    """Make bundled rfmem importable for its internal absolute imports."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rfmem_root = os.path.join(current_dir, "ICLR2026_RF_Mem")
    rfmem_src = os.path.join(rfmem_root, "RF_mem")

    for path in (rfmem_root, rfmem_src):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)

_ensure_rfmem_import_paths()
from .ICLR2026_RF_Mem.RF_mem.personamem_data.retri_mdoel.EmbdRetri import EmbeddingRetrievaler
from .ICLR2026_RF_Mem.RF_mem.personamem_data.utils import decide_strategy_with_probe

class RFMemModel(BaseModel):
    """
    Adapter wrapper for RF-Mem retrieval code.

    Design goals:
    - Keep RF-Mem source code untouched.
    - Reuse RF-Mem methods directly (no method rewrite).
    - Expose lifecycle hooks aligned with project agents:
      __init__ / create_client / begin_dialog / generate / end_dialog.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        retrieve_k: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, **kwargs)

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        # Retrieval knobs (converged to PersonaMem paper/original script defaults).
        self.retrieve_k = int(kwargs.get("retrieve_k", retrieve_k))
        self.probe_k = int(kwargs.get("probe_k", 10))
        self.probe_tau = float(kwargs.get("probe_tau", 0.0))
        self.entropy_th = float(kwargs.get("entropy_th", 0.20))
        self.score_high = float(kwargs.get("score_high", 0.60))
        self.score_low = float(kwargs.get("score_low", 0.30))
        self.retrieve_min_score = float(kwargs.get("retrieve_min_score", 0.30))
        self.beam_width = int(kwargs.get("beam_width", 3))
        self.fanout = int(kwargs.get("fanout", 2))
        self.alpha = float(kwargs.get("alpha", 0.8))
        self.mmr_lambda = float(kwargs.get("mmr_lambda", 0.95))
        self.depth = int(kwargs.get("depth", self.retrieve_k))
        self.rfmem_temperature = float(kwargs.get("rfmem_temperature", 0.7))
        self.rfmem_max_tokens = int(kwargs.get("rfmem_max_tokens", 1024))
        self.rfmem_top_p = float(kwargs.get("rfmem_top_p", 0.9))
        rfmem_seed = kwargs.get("rfmem_seed", 42)
        self.rfmem_seed: Optional[int] = int(rfmem_seed) if rfmem_seed is not None else None

        self.embedding_model_name = str(
            kwargs.get("embedding_model_name", "sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
        )

        self.EmbeddingRetrievaler = EmbeddingRetrievaler
        self.decide_strategy_with_probe = decide_strategy_with_probe

        self.save_agent_logs = bool(kwargs.get("save_agent_logs", True))
        self.logs_output_dir = Path(kwargs.get("agent_logs_output_dir", "agent_logs"))
        self.cache_root = self.logs_output_dir.parent / "rfmem_cache"
        self.cache_root.mkdir(parents=True, exist_ok=True)

        if self.save_agent_logs:
            self.logs_output_dir.mkdir(parents=True, exist_ok=True)

        self._dialog_states: Dict[int, Dict[str, Any]] = {}
        self._state_lock = threading.Lock()

    def _create_rfmem_client(
        self,
        dialog_id: Optional[int],
        dialogue_messages: List[Dict[str, str]],
    ) -> Any:
        cache_tag = f"dialog_{dialog_id}" if dialog_id is not None else "stateless"
        cache_path = str(self.cache_root / cache_tag)
        retriever = self.EmbeddingRetrievaler(
            model_name=self.embedding_model_name,
            top_k=self.retrieve_k,
            cache_path=cache_path,
        )
        if dialogue_messages:
            retriever.build_from_history(dialogue_messages)
        return retriever

    def begin_dialog(self, dialog_id: Optional[int] = None, **kwargs: Any) -> None:
        if dialog_id is None:
            return
        with self._state_lock:
            if dialog_id not in self._dialog_states:
                self._dialog_states[dialog_id] = {
                    "retriever": None,
                    "dialogue_messages": [],  # List[{"role": str, "content": str}]
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
        **kwargs: Any,
    ) -> str:
        dialog_start_time = time.time()
        if not messages:
            return ""

        final_query = ""
        last_user_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if str(messages[i].get("role", "")).lower() == "user":
                final_query = str(messages[i].get("content", "")).strip()
                last_user_idx = i
                break
        if last_user_idx < 0 or not final_query:
            return ""

        dialog_id = kwargs.get("dialog_id")
        if dialog_id is not None:
            with self._state_lock:
                state = self._dialog_states.get(dialog_id)
                if state is None:
                    state = {
                        "retriever": None,
                        "dialogue_messages": [],
                        "last_ingested_idx": 0,
                    }
                    self._dialog_states[dialog_id] = state
        else:
            state = {
                "retriever": None,
                "dialogue_messages": [],
                "last_ingested_idx": 0,
            }

        start_idx = int(state.get("last_ingested_idx", 0))
        if start_idx < 0 or start_idx > last_user_idx:
            start_idx = 0

        dialogue_messages = state["dialogue_messages"]
        new_messages: List[Dict[str, str]] = []
        for i in range(start_idx, last_user_idx):
            msg = messages[i]
            role = str(msg.get("role", "")).lower()
            content = str(msg.get("content", "")).strip()
            if role not in ("user", "assistant") or not content:
                continue
            stamped = {"role": role, "content": content}
            dialogue_messages.append(stamped)
            new_messages.append(stamped)
        state["last_ingested_idx"] = last_user_idx

        new_texts: List[str] = []
        current_user_msg: Optional[str] = None
        for msg in new_messages:
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
                    new_texts.append(f"User: {current_user_msg}\nAssistant: {content}")
                    current_user_msg = None
                else:
                    new_texts.append(f"Assistant: {content}")
        if current_user_msg is not None:
            new_texts.append(f"User: {current_user_msg}")

        retriever = state.get("retriever")
        if retriever is None and dialogue_messages:
            retriever = self._create_rfmem_client(dialog_id=dialog_id, dialogue_messages=dialogue_messages)
            state["retriever"] = retriever
        elif retriever is not None and new_messages:
            try:
                retriever.build_from_history(dialogue_messages)
            except Exception as e:
                logger.warning(
                    "RF-Mem incremental re-build failed for dialog_id=%s: %s; fallback to re-create client",
                    dialog_id,
                    e,
                )
                retriever = self._create_rfmem_client(dialog_id=dialog_id, dialogue_messages=dialogue_messages)
                state["retriever"] = retriever

        retrieved_memories: List[str] = []
        retrieval_mode = "none"
        retriever_ready = False
        if retriever is not None:
            docs = getattr(retriever, "docs", None)
            index = getattr(retriever, "index", None)
            retriever_ready = bool(docs) and index is not None

        if retriever is not None and dialogue_messages and retriever_ready:
            try:
                mode, _diag = self.decide_strategy_with_probe(
                    retriever=retriever,
                    question=final_query,
                    tau=self.probe_tau,
                    probe_k=self.probe_k,
                    ent_th=self.entropy_th,
                    score_th=self.score_high,
                    score_bt=self.score_low,
                )
                retrieval_mode = mode
                if mode == "fast":
                    hits = retriever.retrieve(
                        final_query,
                        top_k=self.retrieve_k,
                        tau=self.retrieve_min_score,
                    )
                else:
                    hits = retriever.retrieve_pyramid_v2(
                        final_query,
                        out_k=self.retrieve_k,
                        tau=self.retrieve_min_score,
                        depth=self.depth,
                        beam_width=self.beam_width,
                        fanout=self.fanout,
                        expansion="mean_group",
                        alpha=self.alpha,
                        mmr_lambda=self.mmr_lambda,
                    )
                retrieved_memories = retriever.to_messages(hits) if hits else []
            except Exception as e:
                logger.warning("RF-Mem retrieval failed for dialog_id=%s: %s", dialog_id, e)

        if retrieved_memories:
            prompt = (
                "Please answer the question based on the relevant chat history.\n\n"
                + "\n".join(str(m) for m in retrieved_memories)
                + "\n\n"
                + final_query
            )
        else:
            prompt = final_query

        effective_temperature = float(kwargs.get("rfmem_temperature", self.rfmem_temperature))
        effective_max_tokens = int(kwargs.get("rfmem_max_tokens", self.rfmem_max_tokens))
        effective_top_p = float(kwargs.get("rfmem_top_p", self.rfmem_top_p))
        rfmem_seed_override = kwargs.get("rfmem_seed", self.rfmem_seed)
        effective_seed: Optional[int] = int(rfmem_seed_override) if rfmem_seed_override is not None else None

        request_payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": effective_temperature,
            "max_tokens": effective_max_tokens,
            "top_p": effective_top_p,
        }
        if effective_seed is not None:
            request_payload["seed"] = effective_seed

        response = self.client.chat.completions.create(**request_payload)
        if not response.choices:
            response_text = ""
        else:
            content = response.choices[0].message.content
            response_text = (content or "").strip()

        if self.save_agent_logs:
            try:
                payload = {
                    "metadata": {
                        "dialog_id": dialog_id,
                        "turn_index": last_user_idx + 1,
                        "query": final_query,
                        "timestamp": time.time(),
                        "latency_seconds": round(time.time() - dialog_start_time, 3),
                        "model_name": self.model_name,
                        "retrieval_mode": retrieval_mode,
                    },
                    "memory_update": {
                        "chunked_documents": new_texts,
                    },
                    "retrieval": {
                        "search_queries": [final_query],
                        "retrieved_contexts": retrieved_memories,
                    },
                    "generation": {
                        "generated_response": response_text,
                    },
                }
                file_name = f"dialog_{dialog_id}.json" if dialog_id is not None else "dialog_stateless.json"
                file_path = self.logs_output_dir / file_name
                if file_path.exists():
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if not isinstance(data, list):
                            data = []
                    except Exception:
                        data = []
                else:
                    data = []
                data.append(payload)
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"Failed to write RF-Mem structured logs: {e}")
        return response_text

