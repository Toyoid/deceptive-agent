"""Build RLHFDataset-compatible parquet files from GRM JSONL."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, List

from .io import read_jsonl
from .schema import GrmJudgeSample


def _validate_rows(rows: Iterable[dict], unlabeled_policy: str) -> List[dict]:
    valid_rows = []
    for idx, row in enumerate(rows):
        sample = GrmJudgeSample.from_dict(row)
        if sample.label is None:
            if unlabeled_policy == "drop":
                continue
            raise ValueError(f"Row {idx} is unlabeled.")
        sample.validate_for_training()
        normalized = sample.to_dict()
        if "split" in row:
            normalized["split"] = row["split"]
        valid_rows.append(normalized)
    return valid_rows


def _split_rows(rows: List[dict], val_ratio: float, seed: int) -> tuple[List[dict], List[dict]]:
    explicit_splits = {str(row.get("split", "")) for row in rows if row.get("split") is not None}
    if explicit_splits:
        train = [row for row in rows if str(row.get("split", "train")) == "train"]
        val = [row for row in rows if str(row.get("split", "")) in {"val", "valid", "validation", "test"}]
        if not train or not val:
            raise ValueError("Explicit split rows must contain both train and val/validation/test rows.")
        return train, val

    shuffled = list(rows)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    val_count = int(round(len(shuffled) * val_ratio))
    if val_ratio > 0.0 and len(shuffled) > 1:
        val_count = max(1, min(len(shuffled) - 1, val_count))
    val = shuffled[:val_count]
    train = shuffled[val_count:]
    return train, val


def _write_parquet(rows: List[dict], path: Path) -> None:
    import datasets

    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = datasets.Dataset.from_list(rows)
    dataset.to_parquet(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert GRM JSONL labels to parquet.")
    parser.add_argument("--input", required=True, help="Labeled GRM JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--unlabeled-policy", choices=["drop", "error"], default="error")
    args = parser.parse_args()

    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be in [0.0, 1.0).")

    rows = _validate_rows(read_jsonl(args.input), args.unlabeled_policy)
    if not rows:
        raise ValueError("No labeled GRM rows available after filtering.")
    train_rows, val_rows = _split_rows(rows, args.val_ratio, args.seed)

    output_dir = Path(args.output_dir)
    _write_parquet(train_rows, output_dir / "train.parquet")
    if val_rows:
        _write_parquet(val_rows, output_dir / "val.parquet")
    print(
        f"[grm build_dataset] train={len(train_rows)} val={len(val_rows)} -> {output_dir}"
    )


if __name__ == "__main__":
    main()
