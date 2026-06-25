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
    rows = []
    path = Path(path)
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
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
