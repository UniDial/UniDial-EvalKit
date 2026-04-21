from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Union

from .schema import Dialog


PathLike = Union[str, Path]


# def _is_probably_meta_file(p: Path) -> bool:
#     # ignore hidden and "private" files (e.g. _meta.json, .DS_Store)
#     return p.name.startswith(".") or p.name.startswith("_")


def iter_jsonl_lines(path: PathLike) -> Iterator[str]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield line


def to_jsonable(obj: Any) -> Any:
    """Best-effort conversion of objects to JSON-serializable structures."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    # Pydantic / SDK response objects often provide model_dump()/dict().
    for attr in ("model_dump", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return to_jsonable(fn())
            except Exception:
                pass

    # Fallback: keep a readable representation to avoid breaking serialization.
    try:
        return {"_repr": repr(obj)}
    except Exception:
        return {"_repr": "<unserializable>"}


# def load_dialogs_from_jsonl(path: PathLike) -> Iterator[Dialog]:
#     for line in iter_jsonl_lines(path):
#         yield Dialog(**json.loads(line))


# def load_dialog_from_json(path: PathLike) -> Dialog:
#     p = Path(path)
#     with p.open("r", encoding="utf-8") as f:
#         return Dialog(**json.load(f))


# def iter_dialog_json_files(dataset_dir: PathLike, *, recursive: bool = False) -> Iterator[Path]:
#     """
#     New dataset format:
#     - one dataset = one folder
#     - one dialog = one JSON file under that folder
#     """
#     root = Path(dataset_dir)
#     if not root.exists():
#         raise FileNotFoundError(root)
#     if not root.is_dir():
#         raise NotADirectoryError(root)

#     pat = "**/*.json" if recursive else "*.json"
#     for p in sorted(root.glob(pat)):
#         if p.is_file() and not _is_probably_meta_file(p):
#             yield p


# def load_dialogs_from_dir(dataset_dir: PathLike, *, recursive: bool = False) -> Iterator[Dialog]:
#     for p in iter_dialog_json_files(dataset_dir, recursive=recursive):
#         yield load_dialog_from_json(p)


# def load_dialogs(path: PathLike, *, recursive: bool = False) -> Iterator[Dialog]:
#     """
#     Backward compatible loader:
#     - if path is a directory: load *.json dialogs
#     - if path endswith .jsonl: load jsonl (one dialog per line)
#     - if path endswith .json: load a single dialog json
#     """
#     p = Path(path)
#     if p.is_dir():
#         yield from load_dialogs_from_dir(p, recursive=recursive)
#         return

#     if not p.exists():
#         raise FileNotFoundError(p)

#     suf = p.suffix.lower()
#     if suf == ".jsonl":
#         yield from load_dialogs_from_jsonl(p)
#     elif suf == ".json":
#         yield load_dialog_from_json(p)
#     else:
#         raise ValueError(f"Unsupported dataset path: {p} (expect .jsonl/.json or a directory)")


# def dump_json(path: PathLike, obj) -> None:
#     p = Path(path)
#     p.parent.mkdir(parents=True, exist_ok=True)
#     with p.open("w", encoding="utf-8") as f:
#         json.dump(obj, f, ensure_ascii=False, indent=2)


# def dumps_jsonl(items: Iterable[dict]) -> str:
#     return "\n".join(json.dumps(x, ensure_ascii=False) for x in items) + "\n"


