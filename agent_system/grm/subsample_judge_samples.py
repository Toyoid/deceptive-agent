"""Subsample dumped GRM judge-sample JSONL files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from .io import read_jsonl, write_jsonl_atomic


def select_one_per_group(rows: List[dict], *, group_size: int = 8, offset: int = 0) -> List[dict]:
    if group_size <= 0:
        raise ValueError("group_size must be positive.")
    if not 0 <= offset < group_size:
        raise ValueError("offset must satisfy 0 <= offset < group_size.")
    return list(rows[offset::group_size])


def _default_output_path(input_path: Path, group_size: int, offset: int) -> Path:
    return input_path.with_name(f"{input_path.stem}_1of{group_size}_offset{offset}{input_path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keep one row from each consecutive group of N dumped GRM judge samples."
    )
    parser.add_argument("--input", required=True, help="Input GRM judge-sample JSONL, e.g. 5.jsonl")
    parser.add_argument("--output", default=None, help="Output JSONL. Defaults to <input>_1ofN_offsetK.jsonl")
    parser.add_argument("--group-size", type=int, default=8, help="Rollouts per original question; default: 8")
    parser.add_argument("--offset", type=int, default=0, help="Which row to keep within each group; default: 0")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output is not None else _default_output_path(
        input_path,
        args.group_size,
        args.offset,
    )
    rows = read_jsonl(input_path)
    selected = select_one_per_group(rows, group_size=args.group_size, offset=args.offset)
    write_jsonl_atomic(output_path, selected)
    print(
        f"[grm subsample] selected {len(selected)}/{len(rows)} rows "
        f"(group_size={args.group_size}, offset={args.offset}) -> {output_path}"
    )


if __name__ == "__main__":
    main()
