from __future__ import annotations

import datetime as dt
import json
import logging
import re
import threading
import time
import uuid
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseModel

logger = logging.getLogger(__name__)


try:
    from .lightmem.memory.lightmem import LightMemory
except ImportError as e:
    logger.error(f"Failed to import LightMem components: {e}")
    traceback.print_exc()
    LightMemory = None


class LightMemModel(BaseModel):
    """
    Wrapper that adapts LightMem to the project's BaseModel interface.

    The model performs:
    1) Incremental ingestion of dialogue history into LightMem.
    2) Retrieval of related memories for the current user query.
    3) Final answer generation via an OpenAI-compatible chat API.
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

        if LightMemory is None:
            raise ImportError("LightMem modules not found.")
        from openai import OpenAI

        self.api_key = api_key or ""
        self.base_url = base_url or ""
        self._lightmem_cls = LightMemory
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.retrieve_k = int(kwargs.get("retrieve_k", retrieve_k))
        self.enable_diagnostics = bool(kwargs.get("lightmem_diagnostics", True))
        self.enable_stage_progress = bool(kwargs.get("lightmem_stage_progress", True))
        self.batch_progress_every = max(1, int(kwargs.get("lightmem_batch_progress_every", 10)))
        self.diagnostic_log_path = Path(
            kwargs.get("lightmem_diag_log_path", "lightmem_agent_errors.jsonl")
        )
        self.save_agent_logs = bool(kwargs.get("save_agent_logs", True))
        self.agent_logs_output_dir = Path(kwargs.get("agent_logs_output_dir", "agent_logs"))
        self.agent_log_max_chars = int(kwargs.get("agent_log_max_chars", 10000))
        if self.save_agent_logs:
            self.agent_logs_output_dir.mkdir(parents=True, exist_ok=True)
        default_device = kwargs.get("default_device")
        if default_device is None:
            default_device = "cpu"
            try:
                import torch  # type: ignore

                if torch.cuda.is_available():
                    default_device = "cuda"
            except Exception:
                default_device = "cpu"

        default_manager = "deepseek" if "deepseek" in str(self.model_name).lower() else "openai"

        # Keep LightMem options in kwargs-driven config (MemoryOS-style),
        # rather than reading from environment variables.
        self.lightmem_config = {
            "pre_compress": kwargs.get("pre_compress", True),
            "topic_segment": kwargs.get("topic_segment", True),
            "precomp_topic_shared": kwargs.get("precomp_topic_shared", True),
            "messages_use": kwargs.get("messages_use", "user_only"),
            "metadata_generate": kwargs.get("metadata_generate", True),
            "text_summary": kwargs.get("text_summary", True),
            "extract_threshold": float(kwargs.get("extract_threshold", 0.1)),
            "index_strategy": kwargs.get("index_strategy", "embedding"),
            "retrieve_strategy": kwargs.get("retrieve_strategy", "embedding"),
            "update": kwargs.get("update", "offline"),
            "embedding_model_name": kwargs.get("embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            "embedding_dims": int(kwargs.get("embedding_dims", 384)),
            "embedding_device": kwargs.get("embedding_device", default_device),
            "llmlingua_model_name": kwargs.get(
                "llmlingua_model_name",
                "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            ),
            "llmlingua_device_map": kwargs.get("llmlingua_device_map", default_device),
            "llmlingua_buffer_len": int(kwargs.get("llmlingua_buffer_len", 512)),
            "compress_rate": float(kwargs.get("compress_rate", 0.6)),
            "shortmem_max_tokens": int(kwargs.get("shortmem_max_tokens", 512)),
            "qdrant_root": kwargs.get("qdrant_root", str(Path("output") / "lightmem_qdrant")),
            "qdrant_on_disk": bool(kwargs.get("qdrant_on_disk", False)),
            "memory_manager_name": kwargs.get("memory_manager_name", default_manager),
            "memory_manager_model": kwargs.get("memory_manager_model", self.model_name),
            "memory_manager_max_tokens": int(kwargs.get("memory_manager_max_tokens", 2048)),
            "memory_manager_temperature": float(kwargs.get("memory_manager_temperature", 0.0)),
            "memory_manager_top_p": float(kwargs.get("memory_manager_top_p", 0.1)),
            "memory_manager_api_key": kwargs.get("memory_manager_api_key", self.api_key),
            "memory_manager_base_url": kwargs.get("memory_manager_base_url", self.base_url),
            "offline_construct_queue_on_end": bool(kwargs.get("offline_construct_queue_on_end", False)),
            "offline_consolidate_on_end": bool(kwargs.get("offline_consolidate_on_end", True)),
            "offline_update_top_k": int(kwargs.get("offline_update_top_k", 20)),
            "offline_update_keep_top_n": int(kwargs.get("offline_update_keep_top_n", 10)),
            "offline_update_workers": int(kwargs.get("offline_update_workers", 4)),
            "offline_update_score_threshold": float(kwargs.get("offline_update_score_threshold", 0.8)),
        }

        self._dialog_states: Dict[int, Dict[str, Any]] = {}
        self._state_lock = threading.Lock()

    def _append_diag_error(self, payload: Dict[str, Any]) -> None:
        """Best-effort persistent error logging for failure localization."""
        try:
            self.diagnostic_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.diagnostic_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write LightMem diagnostics: {e}")

    def _clip_text(self, text: str) -> str:
        if len(text) <= self.agent_log_max_chars:
            return text
        return text[: self.agent_log_max_chars] + "...<truncated>"

    def _dedupe_role_prefix(self, text: str) -> str:
        """Fix duplicated prefixes like 'User: User:' for logging."""
        cleaned = str(text or "").strip()
        cleaned = re.sub(
            r"^(User|Assistant)\s*:\s*(User|Assistant)\s*:\s*",
            lambda m: f"{m.group(2)}: ",
            cleaned,
            flags=re.IGNORECASE,
        )
        return cleaned

    def _sanitize_ingest_content(self, role: str, content: str) -> str:
        """
        Normalize message text before ingestion:
        1) remove duplicated prefixes (User: User:, Assistant: Assistant:)
        2) remove one leading role prefix matching the current role
        """
        cleaned = self._dedupe_role_prefix(content)
        role_prefix = "user" if role == "user" else "assistant"
        cleaned = re.sub(rf"^{role_prefix}\s*:\s*", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def _normalize_text_key(self, text: str) -> str:
        return " ".join(str(text or "").strip().lower().split())

    def _parse_extraction_output(self, output_prompt: str) -> List[Dict[str, Any]]:
        """Parse extraction LLM output into structured raw extraction blocks."""
        text = str(output_prompt or "").strip()
        if not text:
            return []

        sections: List[tuple[str, str]] = []
        factual_key = "Factual:"
        relational_key = "\nRelational:"
        if factual_key in text and relational_key in text:
            factual_raw = text.split(factual_key, 1)[1].split(relational_key, 1)[0].strip()
            relational_raw = text.split(relational_key, 1)[1].strip()
            sections = [("factual", factual_raw), ("relational", relational_raw)]
        else:
            sections = [("factual_or_flat", text)]

        parsed_sections: List[Dict[str, Any]] = []
        pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        for entry_type, raw_text in sections:
            match = re.search(pattern, raw_text)
            cleaned = match.group(1).strip() if match else raw_text.strip()
            parsed_items: List[Any] = []
            try:
                parsed_obj = json.loads(cleaned)
                if isinstance(parsed_obj, dict) and isinstance(parsed_obj.get("data"), list):
                    parsed_items = parsed_obj["data"]
                elif isinstance(parsed_obj, list):
                    parsed_items = parsed_obj
                elif isinstance(parsed_obj, dict):
                    parsed_items = [parsed_obj]
            except Exception:
                parsed_items = []

            parsed_sections.append(
                {
                    "entry_type": entry_type,
                    "raw_output": self._clip_text(raw_text),
                    "parsed_items": parsed_items,
                    "parsed_count": len(parsed_items),
                }
            )
        return parsed_sections

    def _collect_raw_extractions(self, add_result: Dict[str, Any], batch_idx: int) -> List[Dict[str, Any]]:
        """Collect minimal raw extraction facts (no giant prompts)."""
        facts: List[Dict[str, Any]] = []
        output_prompts = add_result.get("add_output_prompt", []) or []
        for call_idx, output_prompt in enumerate(output_prompts):
            parsed_sections = self._parse_extraction_output(str(output_prompt))
            for section_idx, section in enumerate(parsed_sections):
                parsed_items = section.get("parsed_items", [])
                if not isinstance(parsed_items, list):
                    continue
                for item in parsed_items:
                    if not isinstance(item, dict):
                        continue
                    text = str(
                        item.get("fact")
                        or item.get("relation")
                        or item.get("memory")
                        or ""
                    ).strip()
                    if not text:
                        continue
                    facts.append(
                        {
                            "source_id": item.get("source_id"),
                            "text": self._dedupe_role_prefix(text),
                            "batch_idx": batch_idx,
                            "api_call_idx": call_idx,
                            "section_idx": section_idx,
                        }
                    )
        return facts

    def _collect_normalized_entries(self, lightmem: Any, state: Dict[str, Any]) -> tuple[List[Dict[str, Any]], int]:
        if not hasattr(lightmem, "embedding_retriever"):
            return [], 0
        try:
            all_entries = lightmem.embedding_retriever.get_all()
        except Exception as e:
            logger.warning(f"Failed to collect LightMem entries: {e}")
            return [], 0

        if not isinstance(all_entries, list):
            return [], 0

        known_ids = state.setdefault("logged_entry_ids", set())
        if not isinstance(known_ids, set):
            known_ids = set()
            state["logged_entry_ids"] = known_ids

        new_entries: List[Dict[str, Any]] = []
        for entry in all_entries:
            if not isinstance(entry, dict):
                continue
            entry_id = entry.get("id")
            if not entry_id or entry_id in known_ids:
                continue
            known_ids.add(entry_id)
            payload = entry.get("payload", {}) if isinstance(entry.get("payload"), dict) else {}
            new_entries.append(
                {
                    "id": entry_id,
                    "time_stamp": payload.get("time_stamp"),
                    "float_time_stamp": payload.get("float_time_stamp"),
                    "weekday": payload.get("weekday"),
                    "topic_id": payload.get("topic_id"),
                    "memory": self._dedupe_role_prefix(str(payload.get("memory", ""))),
                    "speaker_id": payload.get("speaker_id"),
                    "speaker_name": payload.get("speaker_name"),
                }
            )
        return new_entries, len(all_entries)

    def _collect_entries_by_ids(self, lightmem: Any, entry_ids: List[str]) -> List[Dict[str, Any]]:
        if not entry_ids:
            return []
        if not hasattr(lightmem, "embedding_retriever"):
            return []
        try:
            all_entries = lightmem.embedding_retriever.get_all()
        except Exception as e:
            logger.warning(f"Failed to fetch LightMem entries by ids: {e}")
            return []
        if not isinstance(all_entries, list):
            return []
        wanted = set(entry_ids)
        selected: List[Dict[str, Any]] = []
        for entry in all_entries:
            if not isinstance(entry, dict):
                continue
            entry_id = entry.get("id")
            if entry_id not in wanted:
                continue
            payload = entry.get("payload", {}) if isinstance(entry.get("payload"), dict) else {}
            selected.append(
                {
                    "id": entry_id,
                    "time_stamp": payload.get("time_stamp"),
                    "topic_id": payload.get("topic_id"),
                    "memory": self._dedupe_role_prefix(str(payload.get("memory", ""))),
                }
            )
        return selected

    def _build_indexed_memories(
        self,
        normalized_entries: List[Dict[str, Any]],
        raw_extractions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Join normalized entries with their best-effort extraction source."""
        source_map: Dict[str, List[Dict[str, Any]]] = {}
        for fact in raw_extractions:
            key = self._normalize_text_key(str(fact.get("text", "")))
            if not key:
                continue
            source_map.setdefault(key, []).append(fact)

        indexed_memories: List[Dict[str, Any]] = []
        for entry in normalized_entries:
            memory = self._dedupe_role_prefix(str(entry.get("memory", "")))
            key = self._normalize_text_key(memory)
            candidate_list = source_map.get(key, [])
            matched = candidate_list.pop(0) if candidate_list else None
            turn_id = None
            if matched is not None:
                try:
                    turn_id = int(matched.get("source_id"))
                except Exception:
                    turn_id = None
            topic_id = entry.get("topic_id")
            try:
                topic_id = int(topic_id) if topic_id is not None else None
            except Exception:
                topic_id = None
            indexed_memories.append(
                {
                    "memory": memory,
                    "time_stamp": entry.get("time_stamp"),
                    "ids": {
                        "turn_id": turn_id,
                        "topic_id": topic_id,
                    },
                }
            )
        return indexed_memories

    def _append_structured_dialog_log(self, dialog_id: Optional[int], payload: Dict[str, Any]) -> None:
        if not self.save_agent_logs:
            return
        try:
            self.agent_logs_output_dir.mkdir(parents=True, exist_ok=True)
            file_name = f"dialog_{dialog_id}.json" if dialog_id is not None else "dialog_stateless.json"
            file_path = self.agent_logs_output_dir / file_name
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
            logger.warning(f"Failed to write LightMem structured logs: {e}")

    def _emit_progress(self, dialog_id: Optional[int], msg: str) -> None:
        """Emit progress to both stdout and logger for visibility."""
        if not self.enable_stage_progress:
            return
        line = f"[LightMem][progress] dialog_id={dialog_id} {msg}"
        # Use stdout to avoid logger-level/config swallowing progress lines.
        print(line, flush=True)
        logger.info(line)

    def _create_lightmem_client(self, dialog_id: int) -> Dict[str, Any]:
        init_start = time.perf_counter()
        self._emit_progress(dialog_id, "init_lightmem_client start")
        collection_name = f"dialog_{dialog_id}_{uuid.uuid4().hex[:8]}"
        collection_path = str(Path(self.lightmem_config["qdrant_root"]) / collection_name)
        config = {
            "pre_compress": self.lightmem_config["pre_compress"],
            "pre_compressor": {
                "model_name": "llmlingua-2",
                "configs": {
                    "llmlingua_config": {
                        "model_name": self.lightmem_config["llmlingua_model_name"],
                        "device_map": self.lightmem_config["llmlingua_device_map"],
                        "use_llmlingua2": True,
                    },
                    "compress_config": {
                        "instruction": "",
                        "rate": self.lightmem_config["compress_rate"],
                        "target_token": -1,
                    },
                },
            },
            "topic_segment": self.lightmem_config["topic_segment"],
            "precomp_topic_shared": self.lightmem_config["precomp_topic_shared"],
            "topic_segmenter": {
                "model_name": "llmlingua-2",
                "configs": {
                    "model_name": self.lightmem_config["llmlingua_model_name"],
                    "device_map": self.lightmem_config["llmlingua_device_map"],
                    "buffer_len": self.lightmem_config["llmlingua_buffer_len"],
                },
            },
            "messages_use": self.lightmem_config["messages_use"],
            "metadata_generate": self.lightmem_config["metadata_generate"],
            "text_summary": self.lightmem_config["text_summary"],
            "memory_manager": {
                "model_name": self.lightmem_config["memory_manager_name"],
                "configs": {
                    "model": self.lightmem_config["memory_manager_model"],
                    "api_key": self.lightmem_config["memory_manager_api_key"],
                    "openai_base_url": self.lightmem_config["memory_manager_base_url"],
                    "deepseek_base_url": self.lightmem_config["memory_manager_base_url"],
                    "max_tokens": self.lightmem_config["memory_manager_max_tokens"],
                    "temperature": self.lightmem_config["memory_manager_temperature"],
                    "top_p": self.lightmem_config["memory_manager_top_p"],
                },
            },
            "extract_threshold": self.lightmem_config["extract_threshold"],
            "index_strategy": self.lightmem_config["index_strategy"],
            "text_embedder": {
                "model_name": "huggingface",
                "configs": {
                    "model": self.lightmem_config["embedding_model_name"],
                    "embedding_dims": self.lightmem_config["embedding_dims"],
                    "model_kwargs": {"device": self.lightmem_config["embedding_device"]},
                },
            },
            "retrieve_strategy": self.lightmem_config["retrieve_strategy"],
            "embedding_retriever": {
                "model_name": "qdrant",
                "configs": {
                    "collection_name": collection_name,
                    "embedding_model_dims": self.lightmem_config["embedding_dims"],
                    "path": collection_path,
                    "on_disk": self.lightmem_config["qdrant_on_disk"],
                },
            },
            "update": self.lightmem_config["update"],
        }
        self._emit_progress(dialog_id, f"init_lightmem_client from_config start collection={collection_name}")
        lightmem_client = self._lightmem_cls.from_config(config)
        self._emit_progress(
            dialog_id,
            f"init_lightmem_client from_config done elapsed={round(time.perf_counter() - init_start, 3)}s",
        )
        # Keep this runtime-level override in the wrapper so we can tune th
        # without changing lower-level LightMem package files.
        try:
            lightmem_client.shortmem_buffer_manager.max_tokens = self.lightmem_config["shortmem_max_tokens"]
        except Exception as e:
            logger.warning(f"Failed to override shortmem_max_tokens: {e}")
        self._emit_progress(
            dialog_id,
            f"init_lightmem_client ready collection={collection_name} total_elapsed={round(time.perf_counter() - init_start, 3)}s",
        )
        return {
            "lightmem": lightmem_client,
            "last_ingested_idx": 0,
            "pending_user_msg": None,
            "collection_name": collection_name,
            "logged_entry_ids": set(),
        }

    def begin_dialog(self, dialog_id: Optional[int] = None, **kwargs: Any) -> None:
        if dialog_id is None:
            return
        self._emit_progress(dialog_id, "begin_dialog start")
        with self._state_lock:
            if dialog_id not in self._dialog_states:
                self._emit_progress(dialog_id, "begin_dialog creating_state")
                self._dialog_states[dialog_id] = self._create_lightmem_client(dialog_id=dialog_id)
            else:
                self._emit_progress(dialog_id, "begin_dialog reuse_existing_state")
        self._emit_progress(dialog_id, "begin_dialog done")

    def end_dialog(self, dialog_id: Optional[int] = None, **kwargs: Any) -> None:
        if dialog_id is None:
            return
        # Cleanup dialog-scoped state only. Sleep-time update is performed in
        # generate() before retrieval when enabled.
        with self._state_lock:
            self._dialog_states.pop(dialog_id, None)
        self._emit_progress(dialog_id, "end_dialog done")

    def generate(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
        if not messages:
            return ""

        last_user_idx = -1
        final_query = ""
        for i in range(len(messages) - 1, -1, -1):
            if str(messages[i].get("role", "")).lower() == "user":
                last_user_idx = i
                final_query = str(messages[i].get("content", "")).strip()
                break
        if last_user_idx < 0 or not final_query:
            return ""

        dialog_id = kwargs.get("dialog_id")
        if dialog_id is not None:
            with self._state_lock:
                state = self._dialog_states.get(dialog_id)
                if state is None:
                    state = self._create_lightmem_client(dialog_id=dialog_id)
                    self._dialog_states[dialog_id] = state
        else:
            # Stateless fallback
            state = self._create_lightmem_client(dialog_id=-1)

        stage = "init"
        stage_start = time.perf_counter()
        stage_costs: Dict[str, float] = {}
        current_batch_idx = -1
        batches: List[List[Dict[str, Any]]] = []
        merged_user_count = 0
        orphan_assistant_count = 0
        dialog_start_time = time.perf_counter()
        add_memory_api_calls = 0
        raw_extracted_facts: List[Dict[str, Any]] = []
        new_memory_entries: List[Dict[str, Any]] = []
        indexed_memories: List[Dict[str, Any]] = []
        entries_for_log: List[Dict[str, Any]] = []
        total_memory_entries = 0
        retrieved_contexts: List[str] = []

        def _close_stage(name: str) -> None:
            stage_costs[name] = round(time.perf_counter() - stage_start, 3)

        def _progress(msg: str) -> None:
            self._emit_progress(dialog_id, msg)

        try:
            _progress(f"start last_user_idx={last_user_idx} total_messages={len(messages)}")
            stage = "prepare_state"
            lightmem = state["lightmem"]
            start_idx = int(state.get("last_ingested_idx", 0))
            if start_idx < 0 or start_idx > last_user_idx:
                start_idx = 0

            stage = "build_batches"
            stage_start = time.perf_counter()
            pending_user = state.get("pending_user_msg")
            if pending_user is not None and str(pending_user.get("role", "")).lower() != "user":
                pending_user = None
            for i in range(start_idx, last_user_idx):
                msg = messages[i]
                role = str(msg.get("role", "")).lower()
                content = self._sanitize_ingest_content(
                    role=role,
                    content=str(msg.get("content", "")).strip(),
                )
                if role not in ("user", "assistant") or not content:
                    continue

                # Prefer explicit timestamp fields if present, then fallback.
                time_stamp = None
                for key in ("time_stamp", "timestamp", "time", "datetime"):
                    if key in msg and msg[key]:
                        time_stamp = str(msg[key])
                        break
                if time_stamp is None:
                    ts = dt.datetime.now(dt.timezone.utc) + dt.timedelta(milliseconds=i)
                    time_stamp = ts.isoformat()

                stamped = {
                    "role": role,
                    "content": content,
                    "time_stamp": time_stamp,
                }
                if role == "user":
                    if pending_user is None:
                        pending_user = stamped
                    else:
                        # Keep LightMem input in strict [user, assistant] pairs:
                        # merge consecutive user turns into one pending user entry.
                        prev = str(pending_user.get("content", "")).strip()
                        pending_user["content"] = f"{prev}\n{content}" if prev else content
                        merged_user_count += 1
                    continue
                if pending_user is not None:
                    batches.append([pending_user, stamped])
                    pending_user = None
                else:
                    orphan_assistant_count += 1

            state["pending_user_msg"] = pending_user
            _close_stage("build_batches")
            _progress(
                "build_batches done "
                f"start_idx={start_idx} batch_count={len(batches)} "
                f"merged_users={merged_user_count} orphan_assistants={orphan_assistant_count} "
                f"pending_user={pending_user is not None}"
            )

            stage = "add_memory"
            stage_start = time.perf_counter()
            if batches:
                _progress(f"add_memory start total_batches={len(batches)}")
                for idx, batch in enumerate(batches):
                    current_batch_idx = idx
                    is_last = idx == len(batches) - 1
                    add_result = lightmem.add_memory(
                        messages=batch,
                        force_segment=is_last,
                        force_extract=is_last,
                    )
                    if isinstance(add_result, dict):
                        add_memory_api_calls += int(add_result.get("api_call_nums", 0) or 0)
                        raw_extracted_facts.extend(self._collect_raw_extractions(add_result, idx))
                    if (
                        idx == 0
                        or (idx + 1) % self.batch_progress_every == 0
                        or is_last
                    ):
                        _progress(f"add_memory batch={idx + 1}/{len(batches)}")
            else:
                _progress("add_memory skipped no_new_batches")
            _close_stage("add_memory")
            state["last_ingested_idx"] = last_user_idx
            new_memory_entries, total_memory_entries = self._collect_normalized_entries(lightmem, state)
            entries_for_log = new_memory_entries
            _progress(f"add_memory done elapsed={stage_costs.get('add_memory', 0.0)}s")
            _progress(
                f"memory_entries new={len(new_memory_entries)} total={total_memory_entries} "
                f"api_calls={add_memory_api_calls}"
            )

            # Optional sleep-time update before retrieval:
            # run this before each generation so retrieval uses the latest
            # consolidated memories.
            if self.lightmem_config["offline_consolidate_on_end"]:
                stage = "offline_consolidate"
                stage_start = time.perf_counter()
                _progress("offline_consolidate start")
                try:
                    workers = int(self.lightmem_config["offline_update_workers"])
                    lightmem.construct_update_queue_all_entries(
                        top_k=int(self.lightmem_config["offline_update_top_k"]),
                        keep_top_n=int(self.lightmem_config["offline_update_keep_top_n"]),
                        max_workers=workers,
                    )
                    lightmem.offline_update_all_entries(
                        score_threshold=float(self.lightmem_config["offline_update_score_threshold"]),
                        max_workers=workers,
                    )
                except Exception as e:
                    logger.warning(
                        "LightMem pre-answer sleep update failed for dialog_id=%s: %s\n%s",
                        dialog_id,
                        e,
                        traceback.format_exc(),
                    )
                _close_stage("offline_consolidate")
                _progress(
                    f"offline_consolidate done elapsed={stage_costs.get('offline_consolidate', 0.0)}s"
                )
                # Keep logging content aligned to post-update state.
                post_update_entries = self._collect_entries_by_ids(
                    lightmem,
                    [str(e.get("id")) for e in new_memory_entries if e.get("id")],
                )
                if post_update_entries:
                    entries_for_log = post_update_entries

            indexed_memories = self._build_indexed_memories(entries_for_log, raw_extracted_facts)

            stage = "retrieve"
            stage_start = time.perf_counter()
            _progress("retrieve start")
            retrieved = lightmem.retrieve(final_query, limit=self.retrieve_k)
            if isinstance(retrieved, str):
                retrieved_contexts = [
                    self._dedupe_role_prefix(line) for line in retrieved.splitlines() if line.strip()
                ]
            elif isinstance(retrieved, list):
                retrieved_contexts = [
                    self._dedupe_role_prefix(str(x)) for x in retrieved if str(x).strip()
                ]
            else:
                retrieved_contexts = [self._dedupe_role_prefix(str(retrieved))] if retrieved else []
            _close_stage("retrieve")
            _progress(
                "retrieve done "
                f"elapsed={stage_costs.get('retrieve', 0.0)}s "
                f"retrieved={len(retrieved_contexts)}"
            )
            prompt = (
                "Answer the question based on the retrieved memories. "
                "If memories are insufficient, answer concisely with best effort.\n\n"
                f"Question:\n{final_query}\n\n"
                f"Retrieved memories:\n{str(retrieved) if retrieved else ''}"
            )
            stage = "final_llm"
            stage_start = time.perf_counter()
            _progress("final_llm start")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            _close_stage("final_llm")
            _progress(f"final_llm done elapsed={stage_costs.get('final_llm', 0.0)}s")

            if self.enable_diagnostics:
                logger.info(
                    "[LightMem][diag] dialog_id=%s batches=%s merged_users=%s orphan_assistants=%s "
                    "pending_user=%s stage_costs=%s retrieved=%s",
                    dialog_id,
                    len(batches),
                    merged_user_count,
                    orphan_assistant_count,
                    pending_user is not None,
                    stage_costs,
                    len(retrieved_contexts),
                )
            content = response.choices[0].message.content
            if self.save_agent_logs:
                response_text = (content or "").strip()
                structured_payload = {
                    "metadata": {
                        "dialog_id": dialog_id,
                        "turn_index": last_user_idx + 1,
                        "query": final_query,
                        "timestamp": time.time(),
                        "latency_seconds": round(time.perf_counter() - dialog_start_time, 3),
                        "model_name": self.model_name,
                    },
                    "memory_update": {
                        "indexed_memories": indexed_memories,
                    },
                    "retrieval": {
                        "search_queries": [final_query],
                        "retrieved_contexts": retrieved_contexts,
                    },
                    "generation": {
                        "generated_response": response_text,
                    },
                }
                self._append_structured_dialog_log(dialog_id, structured_payload)
            return (content or "").strip()
        except Exception as e:
            payload = {
                "dialog_id": dialog_id,
                "stage": stage,
                "error_type": type(e).__name__,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "batch_idx": current_batch_idx,
                "batch_count": len(batches),
                "message_count": len(messages),
                "last_user_idx": last_user_idx,
                "start_idx": int(state.get("last_ingested_idx", 0)) if isinstance(state, dict) else None,
                "merged_user_count": merged_user_count,
                "orphan_assistant_count": orphan_assistant_count,
                "stage_costs": stage_costs,
                "model_name": self.model_name,
                "base_url": self.base_url,
                "final_query_preview": final_query[:300],
            }
            self._append_diag_error(payload)
            logger.error(
                "LightMem generation error. dialog_id=%s stage=%s error=%s. "
                "Details persisted to %s\n%s",
                dialog_id,
                stage,
                e,
                self.diagnostic_log_path,
                payload["traceback"],
            )
            raise

