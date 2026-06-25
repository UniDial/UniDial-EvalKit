"""
UniDial EvalKit — FastAPI Backend

Endpoints:
    GET  /api/options                                      — available datasets and model types
    GET  /api/results                                      — existing evaluation results in output_{agent}/
    GET  /api/browse/{agent}/{dataset}/{model}/summary     — load summary.json for a result
    GET  /api/browse/{agent}/{dataset}/{model}/dialogs     — list dialog IDs for a result
    GET  /api/browse/{agent}/{dataset}/{model}/dialogs/{id}— full dialog (generated + eval + agent log)
    GET  /api/tasks/status                                 — current task state (HTTP polling fallback)
    GET  /api/tasks/preview                                — most recently generated dialog (during run)
    POST /api/tasks                                        — create and start an evaluation task
    WS   /api/tasks/ws                                     — real-time progress stream (1 s ticks)
    GET  /api/tasks/result                                 — final aggregated result (after completion)

Run with:
    cd <project-root>
    uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

# Make `src.*` importable when running from project root or this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.config import EvalPipelineConfig
from src.dataset import DATASET_REGISTRY

from backend.models import TaskCreateRequest, TaskStatus
from backend.task_runner import TaskRunner

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="UniDial EvalKit API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

runner = TaskRunner()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Single output directory for all runs.
# Layout:
#   output/{dataset}/{base_model}              — plain LLM (openai, ...)
#   output/{dataset}/{agent}-{base_model}      — memory agent
_OUTPUT_DIR = PROJECT_ROOT / "output"

# Memory agents — directory prefix is "{agent}-".
_KNOWN_AGENTS = ("amem", "lightmem", "memoryos", "mempalace", "hipporag")

# Display names (lowercase dir → readable label). "_base" is the synthetic key
# used for plain-LLM rows that have no agent prefix.
_AGENT_DISPLAY: dict[str, str] = {
    "_base":     "Base LLM",
    "amem":      "A-Mem",
    "lightmem":  "LightMem",
    "memoryos":  "MemoryOS",
    "mempalace": "MemPalace",
    "hipporag":  "HippoRAG",
}

# ---------------------------------------------------------------------------
# Static metadata
# ---------------------------------------------------------------------------

_MODEL_TYPES = [
    {"id": "openai",    "label": "OpenAI / vLLM (compatible API)"},
    {"id": "amem",      "label": "A-Mem Agent"},
    {"id": "lightmem",  "label": "LightMem Agent"},
    {"id": "memoryos",  "label": "MemoryOS Agent"},
    {"id": "mempalace", "label": "MemPalace Agent"},
    {"id": "hipporag",  "label": "HippoRAG Agent"},
]

_DATASET_META: dict[str, dict] = {
    "mt_eval":         {"label": "MT-Eval",         "category": "General"},
    "multi_challenge":  {"label": "Multi-Challenge",  "category": "General"},
    "mt_bench_101":    {"label": "MT-Bench 101",    "category": "General"},
    "multi_if":        {"label": "Multi-IF",         "category": "Instruction Following"},
    "locomo":          {"label": "LoCoMo",           "category": "Memory"},
    "longmemeval":     {"label": "LongMemEval",      "category": "Memory"},
    "personamem":      {"label": "PersonaMem",       "category": "Personalization"},
    "mathchat":        {"label": "MathChat",         "category": "Mathematics"},
    "memorycode":      {"label": "MemoryCode",       "category": "Code"},
    "safedialbench":   {"label": "SafeDialBench",    "category": "Safety"},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> object:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _split_model_dir(model_dir: str) -> tuple[str, str]:
    """
    Split a model directory name into (agent, base_model).
    Returns ('_base', model_dir) when there's no known agent prefix.
    """
    for agent in _KNOWN_AGENTS:
        prefix = agent + "-"
        if model_dir.startswith(prefix):
            return agent, model_dir[len(prefix):]
    return "_base", model_dir


def _run_dir(dataset: str, model: str) -> Path:
    """Filesystem path for a single run."""
    return _OUTPUT_DIR / dataset / model


def _agent_logs_dir(dataset: str, model: str) -> Path:
    """Per-run agent log directory (lives inside the run dir)."""
    return _run_dir(dataset, model) / "agent_logs"


def _find_agent_logs(dataset: str, model: str, dialog_id: str) -> list:
    """Load per-turn agent process logs for one dialog, if present."""
    log_path = _agent_logs_dir(dataset, model) / f"dialog_{dialog_id}.json"
    if log_path.exists():
        try:
            return _load_json(log_path)  # type: ignore[return-value]
        except Exception:
            pass
    return []


# ---------------------------------------------------------------------------
# Routes — static catalog
# ---------------------------------------------------------------------------

@app.get("/api/options")
def get_options():
    """Return all available dataset and model options for the config panel."""
    datasets = [
        {"id": key, **_DATASET_META.get(key, {"label": key, "category": "Other"})}
        for key in DATASET_REGISTRY
    ]
    return {"datasets": datasets, "models": _MODEL_TYPES}


# ---------------------------------------------------------------------------
# Routes — browse existing results
# ---------------------------------------------------------------------------

@app.get("/api/results")
def list_existing_results():
    """
    Scan output/{dataset}/{model_dir}/ for evaluation results.
    `model_dir` is either a plain base model name or "{agent}-{base_model}".
    """
    results = []
    if not _OUTPUT_DIR.exists():
        return results

    for dataset_dir in sorted(_OUTPUT_DIR.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            has_summary = (model_dir / "summary.json").exists()
            gen_dir = model_dir / "generated"
            has_generated = gen_dir.exists() and any(gen_dir.glob("*.json"))
            if not has_summary and not has_generated:
                continue

            agent, base_model = _split_model_dir(model_dir.name)
            meta = _DATASET_META.get(
                dataset_dir.name,
                {"label": dataset_dir.name, "category": "Other"},
            )
            results.append({
                "agent":         agent,
                "agent_label":   _AGENT_DISPLAY.get(agent, agent),
                "dataset":       dataset_dir.name,
                "dataset_label": meta["label"],
                "category":      meta.get("category", "Other"),
                "model":         model_dir.name,
                "base_model":    base_model,
                "has_summary":   has_summary,
            })
    return results


@app.get("/api/browse/{agent}/{dataset}/{model}/summary")
def get_browse_summary(agent: str, dataset: str, model: str):
    """Return summary.json for the given dataset/model directory."""
    path = _run_dir(dataset, model) / "summary.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Summary not found")
    return _load_json(path)


@app.get("/api/browse/{agent}/{dataset}/{model}/dialogs")
def list_browse_dialogs(agent: str, dataset: str, model: str):
    """List all available dialog IDs for the given dataset/model."""
    gen_dir = _run_dir(dataset, model) / "generated"
    if not gen_dir.exists():
        return []
    files = sorted(
        gen_dir.glob("*.json"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else float("inf"),
    )
    return [p.stem for p in files]


@app.get("/api/browse/{agent}/{dataset}/{model}/dialogs/{dialog_id}")
def get_browse_dialog(agent: str, dataset: str, model: str, dialog_id: str):
    """
    Return merged dialog data:
      - generated turns      (output/{dataset}/{model}/generated/{id}.json)
      - eval details         (output/{dataset}/{model}/eval_details/{id}.json)
      - agent process logs   (output/{dataset}/{model}/agent_logs/dialog_{id}.json)
    The `agent` path segment is informational; resolution is by `model` dir name.
    """
    run_dir = _run_dir(dataset, model)
    gen_path = run_dir / "generated" / f"{dialog_id}.json"
    if not gen_path.exists():
        raise HTTPException(status_code=404, detail="Dialog not found")

    generated: dict = _load_json(gen_path)  # type: ignore[assignment]

    eval_path = run_dir / "eval_details" / f"{dialog_id}.json"
    eval_details = _load_json(eval_path) if eval_path.exists() else []

    agent_logs = _find_agent_logs(dataset, model, dialog_id)

    return {
        "dialog_id":     generated.get("dialog_id"),
        "dialog_labels": generated.get("dialog_labels", {}),
        "dialog_turns":  generated.get("dialog_turns", []),
        "eval_details":  eval_details,
        "agent_logs":    agent_logs,
    }


# ---------------------------------------------------------------------------
# Routes — task lifecycle
# ---------------------------------------------------------------------------

@app.post("/api/tasks")
def create_task(req: TaskCreateRequest):
    """Start an evaluation task. Returns 409 if one is already running."""
    if runner.status == TaskStatus.running:
        raise HTTPException(
            status_code=409,
            detail="A task is already running. Wait for it to finish.",
        )

    cfg = EvalPipelineConfig(
        dataset=req.dataset,
        raw_data_dir=req.raw_data_dir,
        processed_data_dir=req.processed_data_dir,
        output_dir=req.output_dir,
        model_type=req.model_type,
        model_name=req.model_name,
        api_key=req.api_key,
        base_url=req.base_url,
        judge_model_type=req.judge_model_type,
        judge_model_name=req.judge_model_name,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        parallel=req.parallel,
        do_generation=req.do_generation,
        do_evaluation=req.do_evaluation,
        agg_by_metric=req.agg_by_metric,
        agg_turn_stat=req.agg_turn_stat,
        agg_dialog_stat=req.agg_dialog_stat,
        agg_dataset_level=req.agg_dataset_level,
    )

    runner.start(cfg)
    return {"message": "Task started"}


@app.get("/api/tasks/status")
def get_status():
    """HTTP polling fallback — returns the same snapshot as the WebSocket tick."""
    return runner.get_progress()


@app.get("/api/tasks/preview")
def get_task_preview():
    """Return the most recently generated dialog during a running task."""
    dialogs = runner.get_recent_generated(1)
    return dialogs[0] if dialogs else None


@app.get("/api/tasks/dialog_list")
def get_task_dialog_list():
    """
    Return compact summary (id + labels) for every dialog generated so far.
    Sorted by modification time (oldest first) so the list grows naturally.
    """
    gen_dir = runner._gen_dir
    if not gen_dir or not gen_dir.exists():
        return []
    files = sorted(gen_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    result = []
    for p in files:
        try:
            import json as _json
            d = _json.loads(p.read_text(encoding="utf-8"))
            result.append({
                "dialog_id": d.get("dialog_id"),
                "dialog_labels": d.get("dialog_labels", {}),
            })
        except Exception:
            pass
    return result


@app.websocket("/api/tasks/ws")
async def task_ws(ws: WebSocket):
    """
    WebSocket: pushes a ProgressSnapshot every second.
    Closes automatically when the task reaches 'completed' or 'failed'.
    """
    await ws.accept()
    try:
        while True:
            snapshot = runner.get_progress()
            await ws.send_json(snapshot)

            if snapshot["status"] in (TaskStatus.completed.value, TaskStatus.failed.value):
                break

            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass


@app.get("/api/tasks/result")
def get_result():
    """Return the final aggregated result. Only available after task completes."""
    if runner.status != TaskStatus.completed:
        raise HTTPException(status_code=404, detail="No completed task available.")
    data = runner.get_result()
    if data is None:
        raise HTTPException(status_code=404, detail="Result file not found on disk.")
    return data
