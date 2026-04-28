from __future__ import annotations

import datetime as dt
import json
import logging
import re
import threading
import time
import uuid
import traceback
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseModel
from src.dataset.data_utils import normalize_statement

logger = logging.getLogger(__name__)

def _ensure_lightmem_import_paths() -> None:
    """Make bundled LightMem importable for its internal absolute imports."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    lightmem_root = os.path.join(current_dir, "LightMem")
    lightmem_src = os.path.join(lightmem_root, "src")

    for path in (lightmem_root, lightmem_src):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


_ensure_lightmem_import_paths()


try:
    from .LightMem.src.lightmem.memory.lightmem import LightMemory
except ImportError as e:
    logger.error(f"Failed to import LightMem components: {e}")
    exit(0)
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
        self.save_agent_logs = bool(kwargs.get("save_agent_logs", True))
        self.agent_logs_output_dir = Path(kwargs.get("agent_logs_output_dir", "agent_logs"))
        if self.save_agent_logs:
            self.agent_logs_output_dir.mkdir(parents=True, exist_ok=True)
        # Ensure intermediate artifacts are colocated with `agent_logs_output_dir` by default.
        artifacts_root = self.agent_logs_output_dir.parent
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
        # Defaults here follow `src/model/LightMem/README.md` recommended config_dict.
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
            "qdrant_root": kwargs.get("qdrant_root", str(artifacts_root / "lightmem_qdrant")),
            # LightMem's Qdrant wrapper deletes the local `path` directory when `on_disk=False`.
            # README recommended config persists collections via local path, so default to True.
            "qdrant_on_disk": bool(kwargs.get("qdrant_on_disk", True)),
            # Optional: align with README's `summary_retriever` example.
            "summary_collection_name": kwargs.get("summary_collection_name"),
            "summary_qdrant_root": kwargs.get(
                "summary_qdrant_root", str(artifacts_root / "lighmem_qdrant_summaries")
            ),
            "memory_manager_name": kwargs.get("memory_manager_name", default_manager),
            "memory_manager_model": kwargs.get("memory_manager_model", self.model_name),
            "memory_manager_max_tokens": int(kwargs.get("memory_manager_max_tokens", 16000)),
            "memory_manager_temperature": float(kwargs.get("memory_manager_temperature", 0.1)),
            "memory_manager_top_p": float(kwargs.get("memory_manager_top_p", 0.1)),
            "memory_manager_api_key": kwargs.get("memory_manager_api_key", self.api_key),
            "memory_manager_base_url": kwargs.get("memory_manager_base_url", self.base_url),
            "offline_construct_queue_on_end": bool(kwargs.get("offline_construct_queue_on_end", False)),
            "offline_consolidate_on_end": bool(kwargs.get("offline_consolidate_on_end", True)),
            "offline_update_top_k": int(kwargs.get("offline_update_top_k", 20)),
            "offline_update_keep_top_n": int(kwargs.get("offline_update_keep_top_n", 10)),
            # LightMem defaults: construct queue uses 8 workers; offline update uses 5.
            "offline_construct_workers": int(kwargs.get("offline_construct_workers", 8)),
            "offline_update_workers": int(kwargs.get("offline_update_workers", 5)),
            "offline_update_score_threshold": float(kwargs.get("offline_update_score_threshold", 0.8)),
        }

        self._dialog_states: Dict[int, Dict[str, Any]] = {}
        self._state_lock = threading.Lock()

    def _dedupe_role_prefix(self, text: str) -> str:
        """Fix duplicated prefixes like 'User: User:' for logging."""
        cleaned, _ = normalize_statement(str(text or ""))
        return cleaned

    def _normalize_timestamp(
        self,
        raw_timestamp: Optional[str],
        fallback_ts: dt.datetime,
        offset_ms: int = 0,
    ) -> str:
        """
        Normalize timestamp to ISO-8601.

        - Prefer parsed DATE strings when available.
        - Fall back to provided UTC timestamp.
        - Add a tiny per-message offset to avoid identical stamps in a batch.
        """
        if not isinstance(fallback_ts, dt.datetime):
            fallback_ts = dt.datetime.now(dt.timezone.utc)
        if fallback_ts.tzinfo is None:
            fallback_ts = fallback_ts.replace(tzinfo=dt.timezone.utc)

        fallback_with_offset = fallback_ts + dt.timedelta(milliseconds=offset_ms)
        if not raw_timestamp:
            return fallback_with_offset.isoformat()

        raw_text = str(raw_timestamp).strip()
        if not raw_text:
            return fallback_with_offset.isoformat()

        # 1) Fast path for ISO-like strings.
        iso_candidate = raw_text.replace("Z", "+00:00")
        try:
            parsed = dt.datetime.fromisoformat(iso_candidate)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=dt.timezone.utc)
            return (parsed + dt.timedelta(milliseconds=offset_ms)).isoformat()
        except ValueError:
            pass

        # 2) Parse common natural-language date strings from datasets.
        normalized = re.sub(r"\s+", " ", raw_text).strip()
        normalized = re.sub(r"(?i)\b(am|pm)\b", lambda m: m.group(1).upper(), normalized)
        normalized_no_comma = normalized.replace(",", "")
        candidates = [normalized, normalized_no_comma]
        patterns = (
            "%I:%M %p on %d %B %Y",
            "%I:%M %p on %d %b %Y",
            "%H:%M on %d %B %Y",
            "%H:%M on %d %b %Y",
            "%d %B %Y %I:%M %p",
            "%d %b %Y %I:%M %p",
            "%B %d %Y %I:%M %p",
            "%b %d %Y %I:%M %p",
        )

        for candidate in candidates:
            for pattern in patterns:
                try:
                    parsed = dt.datetime.strptime(candidate, pattern).replace(tzinfo=dt.timezone.utc)
                    return (parsed + dt.timedelta(milliseconds=offset_ms)).isoformat()
                except ValueError:
                    continue

        return fallback_with_offset.isoformat()

    def _normalize_text_key(self, text: str) -> str:
        return " ".join(str(text or "").strip().lower().split())

    def _format_memories_for_prompt(self, memories: List[str]) -> str:
        """Render retrieved memories with clear separators for readability."""
        cleaned_memories = [self._dedupe_role_prefix(str(m)).strip() for m in memories if str(m).strip()]
        if not cleaned_memories:
            return "(No related memories retrieved.)"
        blocks = []
        for idx, memory in enumerate(cleaned_memories, 1):
            blocks.append(f"[Memory {idx}]\n{memory}")
        return "\n---\n".join(blocks)

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
                    "raw_output": raw_text,
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

    def _create_lightmem_client(self, dialog_id: int) -> Dict[str, Any]:
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

        # Optional: enable summary retriever as in LightMem README example.
        summary_collection_name = self.lightmem_config.get("summary_collection_name")
        if summary_collection_name:
            summary_collection_path = str(
                Path(self.lightmem_config["summary_qdrant_root"]) / str(summary_collection_name)
            )
            config["summary_retriever"] = {
                "model_name": "qdrant",
                "configs": {
                    "collection_name": str(summary_collection_name),
                    "embedding_model_dims": self.lightmem_config["embedding_dims"],
                    "path": summary_collection_path,
                    "on_disk": self.lightmem_config["qdrant_on_disk"],
                },
            }
        lightmem_client = self._lightmem_cls.from_config(config)
        # Keep this runtime-level override in the wrapper so we can tune th
        # without changing lower-level LightMem package files.
        try:
            lightmem_client.shortmem_buffer_manager.max_tokens = self.lightmem_config["shortmem_max_tokens"]
        except Exception as e:
            logger.warning(f"Failed to override shortmem_max_tokens: {e}")
        return {
            "lightmem": lightmem_client,
            "last_ingested_idx": 0,
            "collection_name": collection_name,
            "logged_entry_ids": set(),
            "current_timestamp": None,
        }

    def begin_dialog(self, dialog_id: Optional[int] = None, **kwargs: Any) -> None:
        if dialog_id is None:
            return
        with self._state_lock:
            if dialog_id not in self._dialog_states:
                self._dialog_states[dialog_id] = self._create_lightmem_client(dialog_id=dialog_id)

    def end_dialog(self, dialog_id: Optional[int] = None, **kwargs: Any) -> None:
        if dialog_id is None:
            return
        # Cleanup dialog-scoped state only. Sleep-time update is performed in
        # generate() before retrieval when enabled.
        with self._state_lock:
            self._dialog_states.pop(dialog_id, None)

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
                final_query, _ = normalize_statement(messages[i].get("content", ""))
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

        batches: List[List[Dict[str, Any]]] = []
        dialog_start_time = time.perf_counter()
        raw_extracted_facts: List[Dict[str, Any]] = []
        new_memory_entries: List[Dict[str, Any]] = []
        indexed_memories: List[Dict[str, Any]] = []
        entries_for_log: List[Dict[str, Any]] = []
        retrieved_contexts: List[str] = []

        try:
            lightmem = state["lightmem"]
            start_idx = int(state.get("last_ingested_idx", 0))
            if start_idx < 0 or start_idx > last_user_idx:
                start_idx = 0

            # Build strict [user, assistant] pairs for LightMem ingestion.
            # Compared with the previous logic, we also keep:
            # - first system prompt (as user text),
            # - assistant-only turns,
            # - dangling/continuous user turns via empty assistant fallback.
            current_user_msg = None
            for i in range(start_idx, last_user_idx):
                msg = messages[i]
                role = str(msg.get("role", "")).lower()
                content, date_str = normalize_statement(msg.get("content", ""))

                if role in ("system", "user", "assistant") and content:
                    if date_str is not None:
                        state["current_timestamp"] = date_str
                    ts = dt.datetime.now(dt.timezone.utc)
                    time_stamp = self._normalize_timestamp(
                        state.get("current_timestamp"),
                        fallback_ts=ts,
                        offset_ms=i,
                    )

                    stamped = {
                        "role": role,
                        "content": content,
                        "time_stamp": time_stamp,
                    }

                    if role == "system":
                        if i == 0:
                            system_as_user = {
                                "role": "user",
                                "content": f"{content}",
                                "time_stamp": time_stamp,
                            }
                            batches.append(
                                [
                                    system_as_user,
                                    {"role": "assistant", "content": "", "time_stamp": time_stamp},
                                ]
                            )
                    elif role == "user":
                        if current_user_msg is None:
                            current_user_msg = stamped
                        else:
                            batches.append(
                                [
                                    current_user_msg,
                                    {"role": "assistant", "content": "", "time_stamp": time_stamp},
                                ]
                            )
                            current_user_msg = stamped
                    else:
                        if current_user_msg is not None:
                            batches.append([current_user_msg, stamped])
                            current_user_msg = None
                        else:
                            batches.append(
                                [
                                    {"role": "user", "content": "", "time_stamp": time_stamp},
                                    stamped,
                                ]
                            )

            if current_user_msg is not None:
                batches.append(
                    [
                        current_user_msg,
                        {
                            "role": "assistant",
                            "content": "",
                            "time_stamp": current_user_msg["time_stamp"],
                        },
                    ]
                )
            # Ingest newly formed batches.
            if batches:
                for idx, batch in enumerate(batches):
                    is_last = idx == len(batches) - 1
                    add_result = lightmem.add_memory(
                        messages=batch,
                        force_segment=is_last,
                        force_extract=is_last,
                    )
                    if isinstance(add_result, dict):
                        raw_extracted_facts.extend(self._collect_raw_extractions(add_result, idx))
            state["last_ingested_idx"] = last_user_idx
            new_memory_entries, _ = self._collect_normalized_entries(lightmem, state)
            entries_for_log = new_memory_entries

            # Optional sleep-time update before retrieval:
            # run this before each generation so retrieval uses the latest
            # consolidated memories.
            if self.lightmem_config["offline_consolidate_on_end"]:
                try:
                    construct_workers = int(self.lightmem_config.get("offline_construct_workers") or 8)
                    update_workers = int(self.lightmem_config.get("offline_update_workers") or 5)
                    lightmem.construct_update_queue_all_entries(
                        top_k=int(self.lightmem_config["offline_update_top_k"]),
                        keep_top_n=int(self.lightmem_config["offline_update_keep_top_n"]),
                        max_workers=construct_workers,
                    )
                    lightmem.offline_update_all_entries(
                        score_threshold=float(self.lightmem_config["offline_update_score_threshold"]),
                        max_workers=update_workers,
                    )
                except Exception as e:
                    logger.warning(
                        "LightMem pre-answer sleep update failed for dialog_id=%s: %s\n%s",
                        dialog_id,
                        e,
                        traceback.format_exc(),
                    )
                # Keep logging content aligned to post-update state.
                post_update_entries = self._collect_entries_by_ids(
                    lightmem,
                    [str(e.get("id")) for e in new_memory_entries if e.get("id")],
                )
                if post_update_entries:
                    entries_for_log = post_update_entries

            indexed_memories = self._build_indexed_memories(entries_for_log, raw_extracted_facts)

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
                
            memories_str = self._format_memories_for_prompt(retrieved_contexts)
            prompt = f""" 
Based on following dialogue history and related memories, please answer final question. Ensure your answer is based on the content discussed in the dialogues. 

# User Memories: 
{memories_str}
             
# Final Question
{final_query}
""".strip()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
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
                
                
                # saving logs
                self.agent_logs_output_dir.mkdir(parents=True, exist_ok=True)
                dialog_file_name = (
                    f"dialog_{dialog_id}.json" if dialog_id is not None else "dialog_stateless.json"
                )
                dialog_file = self.agent_logs_output_dir / dialog_file_name

                if dialog_file.exists():
                    try:
                        with open(dialog_file, "r", encoding="utf-8") as f:
                            dialog_data = json.load(f)
                    except json.JSONDecodeError:
                        dialog_data = []
                else:
                    dialog_data = []

                dialog_data.append(structured_payload)

                with open(dialog_file, "w", encoding="utf-8") as f:
                    json.dump(dialog_data, f, ensure_ascii=False, indent=4)

            return (content or "").strip()
        except Exception:
            logger.exception("LightMem generation error")
            raise

