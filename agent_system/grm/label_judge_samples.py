"""Interactive labeler for GRM judge-sample JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import read_jsonl, write_jsonl_atomic


HUMAN_REVIEWED_FIELD = "human_reviewed"
LABEL_SOURCE_FIELD = "label_source"


def _format_prompt(prompt) -> str:
    lines = []
    for message in list(prompt or []):
        role = message.get("role", "")
        content = message.get("content", "")
        lines.append(f"[{role}]\n{content}")
    return "\n\n".join(lines)


def _is_human_reviewed(row) -> bool:
    return bool(row.get(HUMAN_REVIEWED_FIELD, False))


def _first_unreviewed(rows) -> int:
    for idx, row in enumerate(rows):
        if not _is_human_reviewed(row):
            return idx
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactively label GRM judge samples.")
    parser.add_argument("--data", required=True, help="JSONL file to label in place.")
    parser.add_argument("--start", type=int, default=None, help="Optional zero-based row index.")
    args = parser.parse_args()

    path = Path(args.data)
    rows = read_jsonl(path)
    idx = _first_unreviewed(rows) if args.start is None else int(args.start)
    print(f"[grm label] loaded {len(rows)} rows from {path}")

    while idx < len(rows):
        row = rows[idx]
        valid_tokens = [str(token) for token in row.get("valid_tokens", [])]
        current_label = row.get("label")
        print("\n" + "=" * 100)
        print(f"row {idx + 1}/{len(rows)}")
        print(
            f"valid_tokens={valid_tokens} judge_pred_token={row.get('judge_pred_token')} "
            f"label={current_label} label_source={row.get(LABEL_SOURCE_FIELD)} "
            f"human_reviewed={_is_human_reviewed(row)}"
        )
        print("-" * 100)
        print(_format_prompt(row.get("prompt", [])))
        print("-" * 100)
        answer = input(f"label {valid_tokens}, a=accept current, s=skip, q=quit > ").strip()
        if answer.lower() == "q":
            break
        if answer.lower() == "s" or answer == "":
            idx += 1
            continue
        if answer.lower() == "a":
            if current_label not in valid_tokens:
                print(f"Cannot accept current label {current_label!r}; expected one of {valid_tokens!r}.")
                continue
            row[HUMAN_REVIEWED_FIELD] = True
            write_jsonl_atomic(path, rows)
            print(json.dumps(row, ensure_ascii=False))
            idx += 1
            continue
        if answer not in valid_tokens:
            print(f"Invalid label {answer!r}; expected one of {valid_tokens!r}.")
            continue
        row["label"] = answer
        row[LABEL_SOURCE_FIELD] = "human"
        row[HUMAN_REVIEWED_FIELD] = True
        write_jsonl_atomic(path, rows)
        print(json.dumps(row, ensure_ascii=False))
        idx += 1

    print(f"[grm label] stopped at row {idx + 1 if idx < len(rows) else len(rows)}")


if __name__ == "__main__":
    main()
