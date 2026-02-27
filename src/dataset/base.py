# encoding = "utf-8"
import abc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

from .schema import Dialog, Turn, MetricConfig

PathLike = Union[str, Path]


@dataclass
class BenchmarkContext:
    raw_path: Path
    processed_dir: Path
    dataset_name: str


class BenchmarkDataset(abc.ABC):
    """
    Base class for datasets (benchmarks).
    Each benchmark is responsible for:
    - parsing raw data -> Dialog list (schema-compatible)
    - providing prompt templates / defaults (optional)
    - caching processed dialogs to directory
    """

    benchmark_id: str = "base"

    def __init__(self, *, dataset_name: Optional[str] = None) -> None:
        self.dataset_name = dataset_name or self.benchmark_id

    @classmethod
    def meta_version(cls) -> int:
        # Version control
        return 1

    def prompt_templates(self) -> Dict[str, str]:
        # Provide the prompt template collection for this dataset, e.g. {name: template_str}, for reuse during evaluation or inference.
        pass

    def metric_configs(self) -> Dict[str, Any]:
        # Provide default metric configurations for this dataset, e.g. {metric_name: config}, so the evaluation pipeline can initialize metrics with dataset defaults.
        return {}

    @abc.abstractmethod
    def _normalize_raw_data(self, ctx: BenchmarkContext) -> Iterable[Dialog]:
        """
        Convert raw dataset into schema-compatible Dialog objects.
        Subclasses should implement all dataset-specific parsing/mapping here.
        """
        ...
    
    def get_eval_config_for_turn(self, turn: Turn) -> List[MetricConfig]:
        """
        Get evaluation configuration for a specific turn.
        Subclasses can override this to provide dynamic evaluation configuration.
        """
        return turn.eval_config.metrics

    def preprocess(self, *, raw_path: str, processed_root: str, force: bool = False) -> str:
        raw_p = Path(raw_path)
        out_dir = Path(processed_root) / self.dataset_name
        meta_path = out_dir / "_meta.json"
        # prompt_path = out_dir / "_prompt_templates.json"

        raw_abs = str(raw_p.resolve())
        if out_dir.exists() and meta_path.exists() and not force:
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
            if (
                isinstance(meta, dict)
                and meta.get("version") == self.meta_version()
                and meta.get("raw_path") == raw_abs
                and meta.get("benchmark_id") == self.benchmark_id
            ):
                return str(out_dir)

        out_dir.mkdir(parents=True, exist_ok=True)

        ctx = BenchmarkContext(raw_path=raw_p, processed_dir=out_dir, dataset_name=self.dataset_name)
        used: Dict[str, int] = {}
        count = 0
        for idx, dialog in enumerate(self._normalize_raw_data(ctx)):
            fname = self._write_dialog(out_dir, dialog, idx=idx, used=used)
            count += 1

        self._dump_json(meta_path, {"version": self.meta_version(), "raw_path": raw_abs, "benchmark_id": self.benchmark_id, "count": count})
        # if self.prompt_templates():
        #     self._dump_json(prompt_path, {"templates": self.prompt_templates()})
        return str(out_dir)


    def load_eval_dialogs(self, *, data_root: Optional[str] = None, recursive: bool = False, require_alternative_roles: bool = False) -> Iterable[Dialog]:
        """
        Load dialogs for evaluation:
        - data/{benchmark_id}/
        - one dialog = one JSON file
        """

        # root = Path(data_root) if data_root else Path(__file__).resolve().parents[2] / "data"
        # dataset_dir = root / self.benchmark_id
        dataset_dir = Path(data_root)
        if not dataset_dir.exists():
            raise FileNotFoundError(dataset_dir)
        if not dataset_dir.is_dir():
            raise NotADirectoryError(dataset_dir)

        pat = "**/*.json" if recursive else "*.json"
        for p in sorted(dataset_dir.glob(pat)):
            if not p.is_file() or p.name.startswith(".") or p.name.startswith("_"):
                continue
            with p.open("r", encoding="utf-8") as f:
                dialog = Dialog(**json.load(f))
                if require_alternative_roles:
                    dialog = self._ensure_alternating_roles(dialog)
                yield dialog

    
    @staticmethod
    def _ensure_alternating_roles(dialog: Dialog) -> Dialog:
        """
        Ensure dialog turns strictly alternate: user → assistant → user → assistant → ...
        A leading "system" turn is preserved as-is before the alternation starts.
        If two consecutive turns share the same role, insert an empty Turn
        with the missing role in between.
        
        Original turn_ids are preserved to avoid breaking reference_document references.
        Inserted placeholder turns use turn_id=-1.
        """
        if not dialog.dialog_turns:
            return dialog

        fixed_turns: List[Turn] = []
        turns_iter = iter(dialog.dialog_turns)

        # Preserve leading system turn(s)
        for turn in turns_iter:
            if turn.role == "system":
                fixed_turns.append(turn)
            else:
                # Put the first non-system turn back for processing
                remaining = [turn] + list(turns_iter)
                break
        else:
            # All turns are system turns
            return dialog

        expected_role = "user"  # after system, conversation starts with user

        for turn in remaining:
            if turn.role != expected_role:
                # Insert a placeholder turn with turn_id=-1 to avoid breaking references
                fixed_turns.append(Turn(
                    turn_id=-1,
                    role=expected_role,
                    content="",
                ))
                expected_role = "assistant" if expected_role == "user" else "user"

            # Keep the original turn as-is (preserve turn_id)
            fixed_turns.append(turn)
            expected_role = "assistant" if turn.role == "user" else "user"

        dialog = dialog.model_copy(update={"dialog_turns": fixed_turns})
        return dialog

    def _write_dialog(self, out_dir: Path, dialog: Dialog, *, idx: int, used: Dict[str, int]) -> str:
        sid = dialog.dialog_id if dialog.dialog_id is not None else f"dialog_{idx:06d}"
        base = self._safe_filename(str(sid))
        n = used.get(base, 0)
        used[base] = n + 1
        fname = f"{base}.json" if n == 0 else f"{base}__{n}.json"
        self._dump_json(out_dir / fname, dialog.model_dump())
        return fname
    
    @staticmethod
    def _safe_filename(name: str) -> str:
        name = name.strip()
        if not name:
            return "dialog"
        return "".join(c if (c.isalnum() or c in "._-") else "_" for c in name)[:200] or "dialog"
    
    @staticmethod
    def _dump_json(path: PathLike, obj) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)



