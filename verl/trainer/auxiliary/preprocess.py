# Copyright 2026 Hanxiao Li, Beihang University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Preprocess PKU-SafeRLHF prompt-only data and RM-eval pairs."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset


PKU_RESPONSE_COLUMNS = {
    "prompt",
    "response_0",
    "response_1",
    "better_response_id",
    "safer_response_id",
}


def _as_chat_prompt(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def _normalize_prompt(prompt: Any) -> list[dict[str, str]]:
    if isinstance(prompt, str):
        return _as_chat_prompt(prompt)
    if isinstance(prompt, list):
        return [dict(message) for message in prompt]
    if isinstance(prompt, tuple):
        return [dict(message) for message in prompt]
    raise TypeError(f"Expected prompt to be a string or chat-message list, got {type(prompt)}")


def _load_splits(dataset_name: str, subset: str | None = None, split: str | None = None) -> DatasetDict:
    if split is not None:
        dataset = load_dataset(dataset_name, subset, split=split) if subset else load_dataset(dataset_name, split=split)
        return DatasetDict({split: dataset})

    dataset = load_dataset(dataset_name, subset) if subset else load_dataset(dataset_name)
    if isinstance(dataset, Dataset):
        return DatasetDict({"train": dataset})
    return dataset


def _limit_dataset(dataset: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    return dataset.select(range(max_samples))


def _save_dataframe(df: pd.DataFrame, local_dir: str, split: str) -> str:
    os.makedirs(local_dir, exist_ok=True)
    path = os.path.join(local_dir, f"{split}.parquet")
    df.to_parquet(path)
    return path


def build_prompt_only_dataframe(
    dataset: Dataset,
    split: str,
    prompt_key: str,
    data_source: str,
    deduplicate_prompts: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()

    for idx, example in enumerate(dataset):
        prompt = _normalize_prompt(example[prompt_key])

        prompt_key_for_dedup = json.dumps(prompt, ensure_ascii=False, sort_keys=True)
        if deduplicate_prompts and prompt_key_for_dedup in seen_prompts:
            continue
        seen_prompts.add(prompt_key_for_dedup)

        rows.append(
            {
                "data_source": data_source,
                "prompt": prompt,
                "ability": "auxiliary_safety",
                "reward_model": {
                    "style": "model",
                    "ground_truth": None,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "prompt_source": example.get("prompt_source", None),
                },
            }
        )

    return pd.DataFrame(rows)


def build_pku_rm_eval_dataframe(dataset: Dataset, split: str, data_source: str) -> pd.DataFrame:
    missing = PKU_RESPONSE_COLUMNS - set(dataset.column_names)
    if missing:
        raise ValueError(f"Dataset split `{split}` is missing required PKU columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for idx, example in enumerate(dataset):
        prompt = example["prompt"]
        rows.append(
            {
                "data_source": data_source,
                "prompt": _as_chat_prompt(prompt),
                "response_0": example["response_0"],
                "response_1": example["response_1"],
                "better_response_id": int(example["better_response_id"]),
                "safer_response_id": int(example["safer_response_id"]),
                "prompt_source": example.get("prompt_source", None),
                "extra_info": {
                    "split": split,
                    "index": idx,
                },
            }
        )

    return pd.DataFrame(rows)


def preprocess_prompt_only(args: argparse.Namespace) -> None:
    splits = _load_splits(args.dataset, subset=args.subset, split=args.split)
    for split, dataset in splits.items():
        dataset = _limit_dataset(dataset, args.max_samples)
        df = build_prompt_only_dataframe(
            dataset=dataset,
            split=split,
            prompt_key=args.prompt_key,
            data_source=args.data_source,
            deduplicate_prompts=not args.keep_duplicate_prompts,
        )
        path = _save_dataframe(df, args.local_dir, split)
        print(f"[auxiliary preprocess] wrote {len(df)} prompt-only rows to {path}")


def preprocess_pku_rm_eval(args: argparse.Namespace) -> None:
    splits = _load_splits(args.dataset, subset=args.subset, split=args.split)
    for split, dataset in splits.items():
        dataset = _limit_dataset(dataset, args.max_samples)
        df = build_pku_rm_eval_dataframe(dataset=dataset, split=split, data_source=args.data_source)
        path = _save_dataframe(df, args.local_dir, split)
        print(f"[auxiliary preprocess] wrote {len(df)} RM-eval pairs to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess auxiliary prompt-only and PKU RM-eval data.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    prompt_parser = subparsers.add_parser("prompt_only", help="Create RLHFDataset-compatible prompt-only parquet.")
    prompt_parser.add_argument("--dataset", default="PKU-Alignment/PKU-SafeRLHF")
    prompt_parser.add_argument("--subset", default=None)
    prompt_parser.add_argument("--split", default=None, help="Optional single split, e.g. train or test.")
    prompt_parser.add_argument("--local_dir", required=True)
    prompt_parser.add_argument("--prompt_key", default="prompt")
    prompt_parser.add_argument("--data_source", default="pku_safe_rlhf")
    prompt_parser.add_argument("--max_samples", type=int, default=None)
    prompt_parser.add_argument("--keep_duplicate_prompts", action="store_true")
    prompt_parser.set_defaults(func=preprocess_prompt_only)

    rm_parser = subparsers.add_parser("pku_rm_eval", help="Create PKU pairwise parquet for RM accuracy evaluation.")
    rm_parser.add_argument("--dataset", default="PKU-Alignment/PKU-SafeRLHF")
    rm_parser.add_argument("--subset", default=None)
    rm_parser.add_argument("--split", default=None, help="Optional single split, e.g. train or test.")
    rm_parser.add_argument("--local_dir", required=True)
    rm_parser.add_argument("--data_source", default="pku_safe_rlhf")
    rm_parser.add_argument("--max_samples", type=int, default=None)
    rm_parser.set_defaults(func=preprocess_pku_rm_eval)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
