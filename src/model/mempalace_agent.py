from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import threading
import time
from datetime import datetime as _dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from openai import OpenAI

from .base import BaseModel

logger = logging.getLogger(__name__)

try:
    from mempalace.miner import detect_hall, NORMALIZE_VERSION as _NORMALIZE_VERSION
    from mempalace.miner import _extract_entities_for_metadata  # type: ignore
except Exception as e:
    detect_hall = None
    _extract_entities_for_metadata = None
    _NORMALIZE_VERSION = "unknown"
    _mempalace_import_error = e
else:
    _mempalace_import_error = None


# ----- Hybrid boost helpers -----

_STOP_WORDS = {
    "what", "when", "where", "who", "how", "which", "did", "do", "was", "were",
    "have", "has", "had", "is", "are", "the", "a", "an", "my", "me", "i",
    "you", "your", "their", "it", "its", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "ago", "last", "that", "this", "there", "about",
    "get", "got", "give", "gave", "buy", "bought", "made", "make", "said",
}

_NOT_NAMES = {
    "What", "When", "Where", "Who", "How", "Which", "Did", "Do", "Was", "Were",
    "Have", "Has", "Had", "Is", "Are", "The", "My", "Our", "Their",
    "Can", "Could", "Would", "Should", "Will", "Shall", "May", "Might",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "June", "July", "August",
    "September", "October", "November", "December",
    "In", "On", "At", "For", "To", "Of", "With", "By", "From",
    "And", "But", "I", "It", "Its", "This", "That", "These", "Those",
    "Previously", "Recently", "Also", "Just", "Very", "More",
    "Said", "Speaker", "Person", "Time", "Date", "Year", "Day",
}


def _kw(text: str) -> List[str]:
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return [w for w in words if w not in _STOP_WORDS]


def _kw_overlap(query_kws: List[str], doc_text: str) -> float:
    if not query_kws:
        return 0.0
    doc_lower = doc_text.lower()
    hits = sum(1 for kw in query_kws if kw in doc_lower)
    return hits / len(query_kws)


def _quoted_phrases(text: str) -> List[str]:
    phrases: List[str] = []
    for pat in [r"'([^']{3,60})'", r'"([^"]{3,60})"']:
        phrases.extend(re.findall(pat, text))
    return [p.strip() for p in phrases if len(p.strip()) >= 3]


def _quoted_boost(phrases: List[str], doc_text: str) -> float:
    if not phrases:
        return 0.0
    doc_lower = doc_text.lower()
    hits = sum(1 for p in phrases if p.lower() in doc_lower)
    return min(hits / len(phrases), 1.0)


def _person_names(text: str) -> List[str]:
    words = re.findall(r"\b[A-Z][a-z]{2,15}\b", text)
    return list(set(w for w in words if w not in _NOT_NAMES))


def _name_boost(names: List[str], doc_text: str) -> float:
    if not names:
        return 0.0
    doc_lower = doc_text.lower()
    hits = sum(1 for n in names if n.lower() in doc_lower)
    return min(hits / len(names), 1.0)


# ----- Embedding function (fastembed wrapper) -----

_EMBED_MODEL_MAP = {
    "bge-base": "BAAI/bge-base-en-v1.5",
    "bge-large": "BAAI/bge-large-en-v1.5",
    "nomic": "nomic-ai/nomic-embed-text-v1.5",
    "mxbai": "mixedbread-ai/mxbai-embed-large-v1",
}


def _make_embed_fn(model_name: str):
    """Return ChromaDB embedding function or None for default."""
    if not model_name or model_name == "default":
        return None
    hf_name = _EMBED_MODEL_MAP.get(model_name, model_name)
    try:
        import numpy as np  # noqa: F401
        from fastembed import TextEmbedding
        from chromadb.api.types import EmbeddingFunction
    except ImportError as e:
        raise ImportError(
            f"mempalace_embed_model='{model_name}' requires fastembed (and numpy). "
            "Install with: pip install fastembed"
        ) from e

    class _FastEmbedFn(EmbeddingFunction):
        def __init__(self, name: str):
            self._name = name
            self._model = TextEmbedding(model_name=name)

        def __call__(self, input):
            import numpy as _np

            # ChromaDB 1.5+ requires a numpy array (not list[list[float]]).
            return _np.asarray(list(self._model.embed(input)), dtype=_np.float32)

    return _FastEmbedFn(hf_name)


# Suppress chromadb warnings that dump entire embedding vectors on bad upsert.
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.segment").setLevel(logging.ERROR)


class MempalaceModel(BaseModel):
    """
    Thin shell over mempalace + chromadb.

    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, **kwargs)

        if detect_hall is None:
            raise ImportError(
                f"MemPalace components unavailable: {_mempalace_import_error}. "
                "Install/enable mempalace to use model_type='mempalace'."
            )

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        self.mode = str(kwargs.get("mempalace_mode", "hybrid")).lower()
        if self.mode not in {"raw", "hybrid"}:
            raise ValueError(
                f"mempalace_mode must be 'raw' or 'hybrid', got '{self.mode}'"
            )

        self.granularity = str(kwargs.get("mempalace_granularity", "session")).lower()
        if self.granularity not in {"qa_pair", "session"}:
            raise ValueError(
                f"mempalace_granularity must be 'qa_pair' or 'session', got '{self.granularity}'"
            )

        self.top_k = int(kwargs.get("mempalace_top_k", 10))
        self.embed_model = str(kwargs.get("mempalace_embed_model", "bge-large")).strip()
        self.hybrid_weight = float(kwargs.get("mempalace_hybrid_weight", 0.30))

        self.llm_rerank = bool(kwargs.get("mempalace_llm_rerank", self.mode == "hybrid"))
        self.llm_model = str(kwargs.get("mempalace_llm_model") or self.model_name)
        self.llm_base_url = str(kwargs.get("mempalace_llm_base_url", "") or self.base_url or "")
        self.llm_api_key = str(kwargs.get("mempalace_llm_key", "") or self.api_key or "")

        self.default_wing = str(kwargs.get("mempalace_wing", "dialog_memory"))
        self.cache_root = Path(
            kwargs.get("mempalace_cache_root", str(Path("output") / "mempalace_cache"))
        )
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.entry_max_chars = int(kwargs.get("mempalace_entry_max_chars", 1200))

        self.save_agent_logs = bool(kwargs.get("save_agent_logs", True))
        self.logs_output_dir = Path(kwargs.get("agent_logs_output_dir", "agent_logs"))
        if self.save_agent_logs:
            self.logs_output_dir.mkdir(parents=True, exist_ok=True)

        self._dialog_states: Dict[int, Dict[str, Any]] = {}
        self._state_lock = threading.Lock()
        self._embed_fn = _make_embed_fn(self.embed_model)
        self._rerank_client: Optional[OpenAI] = None

    def begin_dialog(self, dialog_id: Optional[int] = None, **kwargs: Any) -> None:
        if dialog_id is None:
            return
        with self._state_lock:
            if dialog_id not in self._dialog_states:
                self._dialog_states[dialog_id] = self._new_state(dialog_id)

    def end_dialog(self, dialog_id: Optional[int] = None, **kwargs: Any) -> None:
        if dialog_id is None:
            return
        with self._state_lock:
            state = self._dialog_states.pop(dialog_id, None)
        if state is None:
            return
        try:
            self._flush_pending(state, force=True)
        except Exception as e:
            logger.warning("flush_pending failed for dialog %s: %s", dialog_id, e)
        shutil.rmtree(state["palace_path"], ignore_errors=True)

    def _new_state(self, dialog_id: Optional[int]) -> Dict[str, Any]:
        tag = f"dialog_{dialog_id}" if dialog_id is not None else f"stateless_{int(time.time() * 1000)}"
        palace_path = self.cache_root / tag
        if palace_path.exists():
            shutil.rmtree(palace_path, ignore_errors=True)
        palace_path.mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(path=str(palace_path))
        collection_kwargs: Dict[str, Any] = {
            "name": "mempalace_drawers",
            "metadata": {"hnsw:space": "cosine"},
        }
        if self._embed_fn is not None:
            collection_kwargs["embedding_function"] = self._embed_fn
        collection = client.get_or_create_collection(**collection_kwargs)

        return {
            "palace_path": str(palace_path),
            "collection": collection,
            "last_ingested_idx": 0,
            "chunk_index": 0,
            "doc_count": 0,
            "qa_acc": [],
            "qa_acc_has_user": False,
            "sess_acc": [],
            "sess_index": 0,
            "qa_index": 0,
            "current_timestamp": None,
            "system_ingested": False,
            "pending_ingested": [],
            "config_written": False,
        }

    # ----- Ingest helpers -----

    @staticmethod
    def _extract_date(content: str) -> Tuple[str, Optional[str]]:
        text = content.lstrip()
        if not text.startswith("DATE:"):
            return content, None
        after_date = text[len("DATE:"):]
        if "\n\n" in after_date:
            date_line, _, rest = after_date.partition("\n\n")
            return rest.strip(), date_line.strip()
        return "", after_date.strip()

    _ROLE_PREFIXES = ("User:", "Assistant:", "user:", "assistant:")

    @classmethod
    def _strip_role_prefix(cls, content: str) -> str:
        text = content.lstrip()
        for p in cls._ROLE_PREFIXES:
            if text.startswith(p):
                rest = text[len(p):]
                return rest[1:] if rest.startswith(" ") else rest
        return content

    @classmethod
    def _render_acc(cls, acc: List[Tuple[str, str]]) -> str:
        lines = []
        for role, content in acc:
            speaker = "User" if role == "user" else "Assistant"
            cleaned = cls._strip_role_prefix(content)
            lines.append(f"{speaker}: {cleaned}")
        return "\n".join(lines)

    @staticmethod
    def _compute_drawer_id(wing: str, room: str, source_file: str, chunk_index: int) -> str:
        """Same drawer_id formula as `mempalace.miner.add_drawer`."""
        h = hashlib.sha256((source_file + str(chunk_index)).encode()).hexdigest()[:24]
        return f"drawer_{wing}_{room}_{h}"

    def _ingest_doc(
        self,
        state: Dict[str, Any],
        doc_text: str,
        source_file: str,
        timestamp: Optional[str] = None,
        corpus_id: Optional[str] = None,
    ) -> None:
        """Inlined `add_drawer` that writes corpus_id + timestamp in one upsert
        (matching `mempalace/benchmarks/locomo_bench.py:738-752`)."""
        if not doc_text or not doc_text.strip():
            return
        if len(doc_text) > self.entry_max_chars:
            doc_text = doc_text[: self.entry_max_chars]
        chunk_index = int(state["chunk_index"])
        wing = self.default_wing
        room = "general"

        drawer_id = self._compute_drawer_id(wing, room, source_file, chunk_index)
        hall = detect_hall(doc_text) if detect_hall is not None else "general"
        entities = ""
        if _extract_entities_for_metadata is not None:
            try:
                entities = _extract_entities_for_metadata(doc_text) or ""
            except Exception:
                entities = ""

        metadata: Dict[str, Any] = {
            "wing": wing,
            "room": room,
            "source_file": source_file,
            "chunk_index": chunk_index,
            "added_by": "evalkit_mempalace",
            "filed_at": _dt.now().isoformat(),
            "normalize_version": _NORMALIZE_VERSION,
            "hall": hall,
        }
        if entities:
            metadata["entities"] = entities
        if corpus_id:
            metadata["corpus_id"] = corpus_id
        if timestamp:
            metadata["timestamp"] = timestamp

        try:
            state["collection"].upsert(
                documents=[doc_text],
                ids=[drawer_id],
                metadatas=[metadata],
            )
        except Exception as e:
            logger.warning("upsert drawer failed: %s", e)
            return

        state.setdefault("pending_ingested", []).append(
            {"doc_id": drawer_id, "hall": hall, "entities": entities}
        )

        state["chunk_index"] = chunk_index + 1
        state["doc_count"] = int(state["doc_count"]) + 1

    @staticmethod
    def _next_qa_corpus_id(state: Dict[str, Any]) -> str:
        idx = int(state.get("qa_index", 0))
        state["qa_index"] = idx + 1
        return f"qa_{idx}"

    @staticmethod
    def _next_session_corpus_id(state: Dict[str, Any]) -> str:
        idx = int(state.get("sess_index", 0))
        return f"session_{idx}"

    def _ingest_qa_pair(self, state: Dict[str, Any], new_msgs: List[Dict[str, str]], source_file: str) -> None:
        """Greedy close on assistant after acc has user; LoCoMo DATE: markers
        are pulled into state['current_timestamp'] and dropped from doc text."""
        acc: List[Tuple[str, str]] = state["qa_acc"]
        acc_has_user: bool = state["qa_acc_has_user"]

        for msg in new_msgs:
            role = str(msg.get("role", "")).lower()
            content = str(msg.get("content", "")).strip()
            if not role or not content:
                continue

            cleaned, date_str = self._extract_date(content)
            if date_str is not None:
                state["current_timestamp"] = date_str
            if not cleaned:
                continue

            acc.append((role, cleaned))
            if role == "user":
                acc_has_user = True
            elif role == "assistant" and acc_has_user:
                self._ingest_doc(
                    state,
                    self._render_acc(acc),
                    source_file,
                    timestamp=state.get("current_timestamp"),
                    corpus_id=self._next_qa_corpus_id(state),
                )
                acc, acc_has_user = [], False

        state["qa_acc"] = acc
        state["qa_acc_has_user"] = acc_has_user

    def _ingest_session(self, state: Dict[str, Any], new_msgs: List[Dict[str, str]], source_file: str) -> None:
        """Split by 'DATE: ' prefix on user turns; flush previous session on boundary."""
        sess_acc: List[Tuple[str, str]] = state["sess_acc"]

        for msg in new_msgs:
            role = str(msg.get("role", "")).lower()
            content = str(msg.get("content", "")).strip()
            if not role or not content:
                continue

            cleaned, date_str = self._extract_date(content)
            is_session_start = date_str is not None

            if is_session_start and sess_acc:
                self._ingest_doc(
                    state,
                    self._render_acc(sess_acc),
                    source_file,
                    timestamp=state.get("current_timestamp"),
                    corpus_id=self._next_session_corpus_id(state),
                )
                sess_acc = []
                state["sess_index"] = int(state["sess_index"]) + 1

            if date_str is not None:
                state["current_timestamp"] = date_str

            if cleaned:
                sess_acc.append((role, cleaned))

        state["sess_acc"] = sess_acc

    def _flush_pending(self, state: Dict[str, Any], force: bool = False) -> None:
        """Flush leftover acc at end_dialog."""
        source_file = "dialog_flush"
        if not force:
            return
        ts = state.get("current_timestamp")

        qa_acc: List[Tuple[str, str]] = state.get("qa_acc", [])
        if qa_acc:
            self._ingest_doc(
                state,
                self._render_acc(qa_acc),
                source_file,
                timestamp=ts,
                corpus_id=self._next_qa_corpus_id(state),
            )
            state["qa_acc"], state["qa_acc_has_user"] = [], False

        sess_acc: List[Tuple[str, str]] = state.get("sess_acc", [])
        if sess_acc:
            self._ingest_doc(
                state,
                self._render_acc(sess_acc),
                source_file,
                timestamp=ts,
                corpus_id=self._next_session_corpus_id(state),
            )
            state["sess_acc"] = []

    # ----- Retrieval & boost -----

    def _query(self, state: Dict[str, Any], query: str, n: int) -> List[Dict[str, Any]]:
        n_docs = int(state.get("doc_count", 0))
        if n_docs <= 0:
            return []
        n_results = max(1, min(n, n_docs))
        try:
            res = state["collection"].query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.warning("collection.query failed: %s", e)
            return []

        docs = (res.get("documents") or [[]])[0] or []
        metas = (res.get("metadatas") or [[]])[0] or []
        dists = (res.get("distances") or [[]])[0] or []

        hits = []
        for doc, meta, dist in zip(docs, metas, dists):
            if doc is None:
                continue
            hits.append(
                {
                    "text": doc,
                    "distance": float(dist),
                    "similarity": round(max(0.0, 1.0 - float(dist)), 4),
                    "metadata": meta or {},
                }
            )
        return hits

    def _apply_hybrid_boost(self, query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not hits:
            return hits
        names = _person_names(query)
        name_words = {n.lower() for n in names}
        all_kws = _kw(query)
        predicate_kws = [w for w in all_kws if w not in name_words]
        quoted = _quoted_phrases(query)

        scored = []
        for hit in hits:
            doc = str(hit.get("text", ""))
            dist = float(hit.get("distance", 1.0))
            pred_overlap = _kw_overlap(predicate_kws, doc)
            fused = dist * (1.0 - 0.50 * pred_overlap)
            q_boost = _quoted_boost(quoted, doc)
            if q_boost > 0:
                fused *= 1.0 - 0.60 * q_boost
            n_boost = _name_boost(names, doc)
            if n_boost > 0:
                fused *= 1.0 - 0.20 * n_boost
            h = dict(hit)
            h["fused_distance"] = round(fused, 4)
            scored.append(h)
        scored.sort(key=lambda h: float(h.get("fused_distance", 1.0)))
        return scored[: self.top_k]

    def _get_rerank_client(self) -> Optional[OpenAI]:
        if self._rerank_client is None:
            try:
                self._rerank_client = OpenAI(
                    api_key=self.llm_api_key or "empty",
                    base_url=self.llm_base_url or None,
                )
            except Exception as e:
                logger.warning("rerank client init failed: %s", e)
                return None
        return self._rerank_client

    def _llm_rerank(self, query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.llm_rerank or len(hits) <= 1:
            return hits
        candidate_cap = min(10, len(hits))
        candidates = hits[:candidate_cap]
        lines = []
        for i, h in enumerate(candidates, 1):
            snippet = str(h.get("text", "")).replace("\n", " ").strip()
            if len(snippet) > 300:
                snippet = snippet[:297] + "..."
            meta = h.get("metadata") or {}
            cid = str(
                meta.get("corpus_id")
                or (f"chunk_{meta.get('chunk_index')}" if meta.get("chunk_index") is not None else "?")
            )
            lines.append(f"{i}. [{cid}] {snippet}")
        prompt = (
            f"Question: {query}\n\n"
            f"Which of the following passages most directly answers this question? "
            f"Reply with just the number (1-{len(candidates)}).\n\n" + "\n".join(lines)
        )

        rerank_client = self._get_rerank_client()
        if rerank_client is None:
            return hits
        try:
            resp = rerank_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=8,
            )
            text = (resp.choices[0].message.content or "").strip() if resp.choices else ""
        except Exception as e:
            logger.warning("LLM rerank failed: %s", e)
            return hits

        m = re.search(r"\b(\d+)\b", text[::-1])
        if not m:
            return hits
        pick = int(m.group(1)[::-1])
        if not 1 <= pick <= len(candidates):
            return hits
        chosen = candidates[pick - 1]
        rest = [h for i, h in enumerate(hits) if i != (pick - 1)]
        return [chosen] + rest

    # ----- Prompt + LLM -----

    @staticmethod
    def _extract_prompt_question(full_text: str) -> str:
        """Drop everything up to and including the FIRST 'Question:' marker.

        For LoCoMo-bundled content this strips the conversational filler and
        the redundant `Question: Based on the above context...` wrapper,
        keeping `Based on...\\nQuestion: <real Q> Short answer:` for the LLM.
        Returned unchanged if no `Question:` marker is present.
        """
        if not full_text or "Question:" not in full_text:
            return full_text
        _, _, rest = full_text.partition("Question:")
        return rest.strip()

    def _build_prompt(self, query: str, hits: List[Dict[str, Any]]) -> str:
        prompt_question = self._extract_prompt_question(query)
        if not hits:
            return f"[Task]\n{prompt_question}"
        snippets = []
        for i, hit in enumerate(hits[: self.top_k], 1):
            text = str(hit.get("text", "")).strip()
            if len(text) > 700:
                text = text[:700] + "...<truncated>"
            meta = hit.get("metadata") or {}
            ts = str(meta.get("timestamp", "") or "").strip()
            header = f"[Memory {i}, {ts}]" if ts else f"[Memory {i}]"
            snippets.append(f"{header} {text}")
        return (
            "Use the following relevant memories if they help answer the question.\n\n"
            + "\n\n".join(snippets)
            + "\n\n[Task]\n"
            + prompt_question
        )

    @staticmethod
    def _extract_retrieval_query(full_text: str) -> str:
        if not full_text:
            return full_text
        if "Question:" not in full_text:
            return full_text

        tail = full_text.rsplit("Question:", 1)[-1].strip()

        for marker in ("Short answer:", "Short answer", "Answer:"):
            if marker in tail:
                tail = tail.split(marker)[0].strip()

        cat2_suffix = "Use DATE of CONVERSATION to answer with an approximate date."
        if cat2_suffix in tail:
            tail = tail.replace(cat2_suffix, "").strip()

        cat5_match = re.search(r"\s*Select the correct answer:.*$", tail, flags=re.DOTALL)
        if cat5_match:
            tail = tail[: cat5_match.start()].strip()

        return tail or full_text

    @staticmethod
    def _system_content(messages: List[Dict[str, Any]]) -> str:
        for msg in messages:
            if str(msg.get("role", "")).lower() == "system":
                cand = str(msg.get("content", "") or "").strip()
                if cand:
                    return cand
        return "You are a helpful assistant."

    # ----- Logging -----

    def _agent_config(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "granularity": self.granularity,
            "top_k": self.top_k,
            "embed_model": self.embed_model,
            "hybrid_weight": self.hybrid_weight,
            "llm_rerank": self.llm_rerank,
            "llm_model": self.llm_model,
            "model_name": self.model_name,
        }

    @staticmethod
    def _hit_to_log(hit: Dict[str, Any]) -> Dict[str, Any]:
        meta = hit.get("metadata") or {}
        keep_keys = (
            "wing", "room", "hall", "entities",
            "source_file", "chunk_index", "corpus_id", "timestamp",
        )
        small_meta = {k: meta.get(k) for k in keep_keys if k in meta}
        return {
            "text": hit.get("text", ""),
            "similarity": hit.get("similarity"),
            "metadata": small_meta,
        }

    def _write_log(
        self,
        state: Dict[str, Any],
        dialog_id: Optional[int],
        turn_index: int,
        query: str,
        hits: List[Dict[str, Any]],
        ingested_docs: List[Dict[str, Any]],
        response_text: str,
        latency_s: float,
    ) -> None:
        if not self.save_agent_logs:
            return
        payload = {
            "metadata": {
                "dialog_id": dialog_id,
                "turn_index": turn_index,
                "query": query,
                "timestamp": time.time(),
                "latency_seconds": round(latency_s, 3),
                "model_name": self.model_name,
            },
            "memory_update": {
                "ingested_docs": list(ingested_docs),
            },
            "retrieval": {
                "search_queries": [query],
                "retrieved_contexts": [self._hit_to_log(h) for h in hits[: self.top_k]],
            },
            "generation": {
                "generated_response": response_text,
            },
        }
        file_name = (
            f"dialog_{dialog_id}.json" if dialog_id is not None else "dialog_stateless.json"
        )
        file_path = self.logs_output_dir / file_name
        try:
            data: List[Any]
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

            if not data and not state.get("config_written"):
                data.append({"agent_config": self._agent_config()})
                state["config_written"] = True
            elif data and not state.get("config_written"):
                state["config_written"] = True

            data.append(payload)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("write_log failed: %s", e)

    # ----- Main entry -----

    def generate(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
        start_time = time.time()
        if not messages:
            return ""

        final_query = ""
        last_user_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if str(messages[i].get("role", "")).lower() == "user":
                final_query = str(messages[i].get("content", "") or "").strip()
                last_user_idx = i
                break
        if last_user_idx < 0 or not final_query:
            return ""

        dialog_id = kwargs.get("dialog_id")
        ephemeral = dialog_id is None
        if ephemeral:
            state = self._new_state(None)
        else:
            with self._state_lock:
                state = self._dialog_states.get(dialog_id) or self._new_state(dialog_id)
                self._dialog_states[dialog_id] = state

        # Phase 1: Ingest. System messages become standalone docs
        # (corpus_id="system") so they don't get rendered as `Assistant: ...`;
        # user/assistant messages flow through qa_pair / session ingestion.
        start_idx = int(state.get("last_ingested_idx", 0))
        if start_idx < 0 or start_idx > last_user_idx:
            start_idx = 0
        new_msgs: List[Dict[str, str]] = []
        new_system_msgs: List[str] = []
        for i in range(start_idx, last_user_idx):
            msg = messages[i]
            role = str(msg.get("role", "")).lower()
            content = str(msg.get("content", "") or "").strip()
            if not content:
                continue
            if role == "system":
                new_system_msgs.append(content)
            elif role in {"user", "assistant"}:
                new_msgs.append({"role": role, "content": content})
        state["last_ingested_idx"] = last_user_idx

        source_file = (
            f"dialog_{dialog_id}" if dialog_id is not None else "dialog_stateless"
        )

        if new_system_msgs and not state.get("system_ingested"):
            self._ingest_doc(
                state, new_system_msgs[0], source_file,
                timestamp=None, corpus_id="system",
            )
            state["system_ingested"] = True

        if new_msgs:
            if self.granularity == "qa_pair":
                self._ingest_qa_pair(state, new_msgs, source_file)
            else:
                self._ingest_session(state, new_msgs, source_file)

        # Use the bare question for retrieval to match MemPalace's bench
        # (which queries with raw `qa["question"]`); the full bundled
        # `final_query` is still used to build the LLM prompt.
        retrieval_query = self._extract_retrieval_query(final_query)

        # Phase 2: Retrieve.
        if self.mode == "raw":
            hits = self._query(state, retrieval_query, n=self.top_k)
        else:
            hits = self._query(state, retrieval_query, n=self.top_k * 3)
            hits = self._apply_hybrid_boost(retrieval_query, hits)

        # Phase 3: LLM rerank (hybrid only).
        if self.mode == "hybrid":
            hits = self._llm_rerank(retrieval_query, hits)

        # Phase 4: Generate.
        prompt = self._build_prompt(final_query, hits)
        system_content = self._system_content(messages)
        effective_temperature = 0.0 if temperature is None else float(temperature)
        effective_max_tokens = 1024 if max_tokens is None else int(max_tokens)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt},
                ],
                temperature=effective_temperature,
                max_tokens=effective_max_tokens,
            )
            content = response.choices[0].message.content if response.choices else ""
            response_text = (content or "").strip()
        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            response_text = ""

        # Phase 5: Log.
        ingested_docs = state.get("pending_ingested", [])
        state["pending_ingested"] = []
        latency_s = time.time() - start_time
        self._write_log(
            state=state,
            dialog_id=dialog_id,
            turn_index=last_user_idx + 1,
            query=retrieval_query,
            hits=hits,
            ingested_docs=ingested_docs,
            response_text=response_text,
            latency_s=latency_s,
        )

        if ephemeral:
            shutil.rmtree(state["palace_path"], ignore_errors=True)

        return response_text
