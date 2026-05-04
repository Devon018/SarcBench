from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

TASK_FILES = {
    "task1": "task1_benchmark.jsonl",
    "task2": "task2_benchmark.jsonl",
    "task3": "task3_benchmark.jsonl",
}


def task_path(data_dir: Path, task: str) -> Path:
    try:
        filename = TASK_FILES[task]
    except KeyError as exc:
        raise ValueError(f"Unknown task {task!r}; expected one of {sorted(TASK_FILES)}") from exc
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing benchmark file: {path}")
    return path


def iter_jsonl(path: Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            count += 1
            if limit is not None and count >= limit:
                break


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    return list(iter_jsonl(path, limit=limit))


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_completed_ids(path: Path, key_fields: tuple[str, ...] = ("sample_id",)) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    for row in iter_jsonl(path):
        values = [str(row.get(field, "")) for field in key_fields]
        if all(values):
            completed.add("||".join(values))
    return completed
