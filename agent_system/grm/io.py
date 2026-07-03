"""JSONL I/O for GRM judge datasets."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List

from .schema import GrmJudgeSample


JUDGE_SAMPLES_FILENAME = "judge_samples.jsonl"
JUDGE_SAMPLES_STEP_TEMPLATE = "{step}.jsonl"


def judge_samples_path(data_dir: str | os.PathLike[str], step: int | None = None) -> Path:
    if step is not None:
        return Path(data_dir) / JUDGE_SAMPLES_STEP_TEMPLATE.format(step=int(step))
    return Path(data_dir) / JUDGE_SAMPLES_FILENAME


def read_jsonl(path: str | os.PathLike[str]) -> List[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GRM JSONL file not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"GRM JSONL path is a directory: {path}")

    text = path.read_text(encoding="utf-8")
    stripped_text = text.strip()
    if not stripped_text:
        raise ValueError(f"GRM JSONL file is empty: {path}")

    try:
        parsed = json.loads(stripped_text)
    except json.JSONDecodeError:
        parsed = None
    else:
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            for idx, row in enumerate(parsed):
                if not isinstance(row, dict):
                    raise ValueError(
                        f"Expected JSON object at array index {idx} in {path}, got {type(row).__name__}."
                    )
            return parsed

    rows = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse JSON on line {line_no} of {path}: {exc.msg}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(
                f"Expected JSON object on line {line_no} of {path}, got {type(row).__name__}."
            )
        rows.append(row)
    if not rows:
        raise ValueError(f"GRM JSONL file has no JSON object rows: {path}")
    return rows


def write_jsonl_atomic(path: str | os.PathLike[str], rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def append_judge_samples(
    data_dir: str | os.PathLike[str],
    samples: Iterable[GrmJudgeSample],
    step: int | None = None,
) -> int:
    path = judge_samples_path(data_dir, step=step)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count
