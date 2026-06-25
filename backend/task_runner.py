"""
TaskRunner — runs EvalPipeline in a background thread and exposes
poll-safe progress information for the WebSocket handler.

Design notes:
- Only one task at a time (MVP constraint).
- Progress is tracked by counting checkpoint files in output dirs;
  no modifications to EvalPipeline or phases are required.
- Logs are captured from the root logger via a queue-backed handler
  and drained by get_progress() on each WebSocket tick.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import EvalPipelineConfig
from src.eval_pipeline import EvalPipeline
from backend.models import TaskStatus


# ---------------------------------------------------------------------------
# Log capture
# ---------------------------------------------------------------------------

class _QueueLogHandler(logging.Handler):
    """Forwards formatted log records into a thread-safe queue."""

    def __init__(self, q: "queue.Queue[str]") -> None:
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.q.put_nowait(self.format(record))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class TaskRunner:
    """Single-task runner. MVP supports one concurrent task."""

    def __init__(self) -> None:
        self.status: TaskStatus = TaskStatus.idle
        self.phase: str = "idle"
        self.total: int = 0
        self.error: Optional[str] = None
        self.result: Optional[Dict[str, Any]] = None

        self._cfg: Optional[EvalPipelineConfig] = None
        self._gen_dir: Optional[Path] = None
        self._eval_dir: Optional[Path] = None
        self._log_q: "queue.Queue[str]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API (called from FastAPI handlers — must be thread-safe)
    # ------------------------------------------------------------------

    def start(self, cfg: EvalPipelineConfig) -> None:
        if self.status == TaskStatus.running:
            raise RuntimeError("A task is already running.")

        # Reset state
        self.status = TaskStatus.running
        self.phase = "data"
        self.total = 0
        self.error = None
        self.result = None
        self._cfg = cfg
        self._gen_dir = cfg.gen_output_dir
        self._eval_dir = cfg.eval_output_dir

        # Drain stale logs from previous run
        while not self._log_q.empty():
            try:
                self._log_q.get_nowait()
            except queue.Empty:
                break

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def get_progress(self) -> Dict[str, Any]:
        """Return a snapshot suitable for JSON serialisation."""
        return {
            "status": self.status.value,
            "phase": self.phase,
            "total": self.total,
            "completed": self._count_completed(),
            "logs": self._drain_logs(),
            "error": self.error,
        }

    def get_result(self) -> Optional[Dict[str, Any]]:
        """Return aggregated result; falls back to reading summary.json from disk."""
        if self.result:
            return self.result
        if self._cfg and self._cfg.summary_output_path.exists():
            try:
                with open(self._cfg.summary_output_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def get_recent_generated(self, n: int = 3) -> List[Dict[str, Any]]:
        """Return the N most-recently written generated-dialog files (intermediate preview)."""
        if not self._gen_dir or not self._gen_dir.exists():
            return []
        files = sorted(
            self._gen_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        dialogs = []
        for p in files[:n]:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    dialogs.append(json.load(f))
            except Exception:
                pass
        return dialogs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _count_completed(self) -> int:
        phase = self.phase
        if phase == "generation" and self._gen_dir and self._gen_dir.exists():
            return len(list(self._gen_dir.glob("*.json")))
        if phase in ("evaluation", "aggregation") and self._eval_dir and self._eval_dir.exists():
            return len(list(self._eval_dir.glob("*.json")))
        if phase == "done":
            return self.total
        return 0

    def _drain_logs(self) -> List[str]:
        logs: List[str] = []
        while True:
            try:
                logs.append(self._log_q.get_nowait())
            except queue.Empty:
                break
        return logs

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        handler = _QueueLogHandler(self._log_q)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
        )
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)

        try:
            cfg = self._cfg
            pipeline = EvalPipeline(cfg)

            # Phase 1: Data
            self.phase = "data"
            dialogs = pipeline.prepare_data()
            self.total = len(dialogs)

            # Phase 2: Generation
            if cfg.do_generation:
                self.phase = "generation"
                generated = pipeline.run_generation(dialogs)
            else:
                generated = pipeline._load_generated_dialogs()

            # Phase 3: Evaluation
            if cfg.do_evaluation:
                if not generated:
                    raise RuntimeError(
                        "No generated dialogs found. "
                        "Enable do_generation or ensure generated files exist on disk."
                    )
                self.phase = "evaluation"
                results = pipeline.run_evaluation(generated)

                # Phase 4: Aggregation
                self.phase = "aggregation"
                self.result = pipeline.run_aggregation(results)

            self.phase = "done"
            self.status = TaskStatus.completed

        except Exception as exc:
            self.error = str(exc)
            self.status = TaskStatus.failed
            logging.getLogger(__name__).exception("Pipeline failed")
        finally:
            root_logger.removeHandler(handler)
