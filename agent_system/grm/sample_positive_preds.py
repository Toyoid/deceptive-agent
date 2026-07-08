"""Subsample dumped GRM judge-sample JSONL files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from agent_system.grm.schema import GrmJudgeSample

from .io import read_jsonl, write_jsonl_atomic

_STRICT5_TOKENS = ("0", "1", "2", "3", "4")
_NEGATIVE_TOKENS = {"0", "1"}
_NEUTRAL_TOKENS = {"2"}
_POSITIVE_TOKENS = {"3", "4"}

def sample_positive_predictions(rows: List[dict]) -> Tuple[List[dict], dict[str, float]]:
    positive_predictions = []
    positive_preds = neutral_preds = negative_preds = 0
    tp = fp = fn = tn = 0
    for row in rows:
        sample = GrmJudgeSample.from_dict(row)
    
        if tuple(sample.valid_tokens) != _STRICT5_TOKENS:
            continue
        if sample.judge_pred_token is None:
            continue
        if sample.judge_pred_token not in _STRICT5_TOKENS:
            continue

        if sample.judge_pred_token in _NEGATIVE_TOKENS:
            negative_preds += 1
        elif sample.judge_pred_token in _NEUTRAL_TOKENS:
            neutral_preds += 1
        elif sample.judge_pred_token in _POSITIVE_TOKENS:
            positive_preds += 1
            positive_predictions.append(row)

        if sample.label is not None and sample.label in _STRICT5_TOKENS:
            if sample.judge_pred_token in _POSITIVE_TOKENS and sample.label in _POSITIVE_TOKENS:
                tp += 1
            elif sample.judge_pred_token in _POSITIVE_TOKENS and sample.label in _NEGATIVE_TOKENS:
                fp += 1
            elif sample.judge_pred_token in _NEGATIVE_TOKENS and sample.label in _POSITIVE_TOKENS:
                fn += 1
            elif sample.judge_pred_token in _NEGATIVE_TOKENS and sample.label in _NEGATIVE_TOKENS:
                tn += 1

    return positive_predictions, {
        "positive_pred_count": positive_preds,
        "neutral_pred_count": neutral_preds,
        "negative_pred_count": negative_preds,
        "tp_ratio": tp / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0,
        "fp_ratio": fp / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0,
        "fn_ratio": fn / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0,
        "tn_ratio": tn / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0,
    }


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_positive_preds{input_path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keep one row from each consecutive group of N dumped GRM judge samples."
    )
    parser.add_argument("--input", required=True, help="Input GRM judge-sample JSONL, e.g. 5.jsonl")
    parser.add_argument("--output", default=None, help="Output JSONL. Defaults to <input>_positive_preds.jsonl")
    parser.add_argument("--check-only", action="store_true", help="Check the input file and exit without writing output")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output is not None else _default_output_path(input_path)
    rows = read_jsonl(input_path)
    selected, stats = sample_positive_predictions(rows)
    if args.check_only:
        print(f"[grm subsample] {len(selected)}/{len(rows)} rows selected as positive predictions")
        print(f"Statistics: {stats}")
        return
    write_jsonl_atomic(output_path, selected)
    print(f"Statistics: {stats}")
    print(
        f"[grm subsample] selected {len(selected)}/{len(rows)} rows as positive predictions -> {output_path}"
    )


if __name__ == "__main__":
    main()
