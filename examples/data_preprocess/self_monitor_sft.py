#!/usr/bin/env python3
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

"""
Preprocess the PKU-Alignment/self-monitor dataset into multi-turn SFT parquet files.
Optionally filter out overlong examples using the same chat-template tokenization
path as training.
"""

import argparse
import json
import os
from collections import Counter
from typing import Iterable, Optional

import numpy as np
import pandas as pd


DEFAULT_HF_REPO_ID = "PKU-Alignment/self-monitor"
DEFAULT_VAL_RATIO = 0.1
DEFAULT_SEED = 1
SYSTEM_PROMPT = (
    "You are a helpful and harmless assistant.\n"
    "Before answering, think step by step, and your response must follow this format:\n"
    "<think>\n"
    "your reasoning here\n"
    "</think>\n"
    "your final answer"
)
USER_FORMAT_PROMPT = (
    "Think step by step before answering, and respond exactly in this format:\n"
    "<think>\n"
    "your reasoning here\n"
    "</think>\n"
    "your final answer"
)
REQUIRED_FIELDS = ("question", "thinking_process", "reflection", "assessment", "revised_response")
RESERVED_CLOSING_TAGS = ("</think>", "</monitor>", "</label>")


def normalize_required_text(value) -> Optional[str]:
    if value is None:
        return None
    if pd.isna(value):
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    return normalized


def find_reserved_tag(text: str) -> Optional[str]:
    for tag in RESERVED_CLOSING_TAGS:
        if tag in text:
            return tag
    return None


def build_target_text(thinking_process: str, reflection: str, assessment: str, revised_response: str) -> str:
    return (
        "<think>\n"
        f"{thinking_process}\n"
        f"<monitor>{reflection}</monitor>\n"
        f"<label>{assessment}</label>\n"
        "</think>\n"
        f"{revised_response}"
    )


def build_user_prompt(question: str, format_prompt: str = USER_FORMAT_PROMPT) -> str:
    format_prompt = format_prompt.strip()
    if not format_prompt:
        return question
    return f"{question}\n\n{format_prompt}"


def build_messages(user_prompt: str, target_text: str, system_prompt: str = SYSTEM_PROMPT) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": target_text},
    ]


def process_self_monitor_row(row: dict, system_prompt: str = SYSTEM_PROMPT) -> tuple[Optional[dict], Optional[str]]:
    normalized_fields = {}
    for field in REQUIRED_FIELDS:
        normalized = normalize_required_text(row.get(field))
        if normalized is None:
            return None, f"missing_or_empty:{field}"
        reserved_tag = find_reserved_tag(normalized)
        if reserved_tag is not None:
            return None, f"reserved_tag:{reserved_tag}"
        normalized_fields[field] = normalized

    target_text = build_target_text(
        thinking_process=normalized_fields["thinking_process"],
        reflection=normalized_fields["reflection"],
        assessment=normalized_fields["assessment"],
        revised_response=normalized_fields["revised_response"],
    )
    user_prompt = build_user_prompt(normalized_fields["question"])

    processed_row = dict(row)
    processed_row.update(normalized_fields)
    processed_row["system_prompt"] = system_prompt
    processed_row["user_prompt"] = user_prompt
    processed_row["target_text"] = target_text
    processed_row["messages"] = build_messages(
        user_prompt=user_prompt,
        target_text=target_text,
        system_prompt=system_prompt,
    )
    return processed_row, None


def prepare_self_monitor_dataframe(df: pd.DataFrame, system_prompt: str = SYSTEM_PROMPT) -> tuple[pd.DataFrame, Counter]:
    processed_rows = []
    drop_reasons: Counter = Counter()

    for row in df.to_dict(orient="records"):
        processed_row, drop_reason = process_self_monitor_row(row=row, system_prompt=system_prompt)
        if drop_reason is not None:
            drop_reasons[drop_reason] += 1
            continue
        processed_rows.append(processed_row)

    processed_df = pd.DataFrame(processed_rows)
    return processed_df, drop_reasons


def load_chat_tokenizer(tokenizer_name_or_path: str):
    from verl.utils.tokenizer import hf_tokenizer

    return hf_tokenizer(tokenizer_name_or_path)


def compute_message_sequence_length(messages: list[dict[str, str]], tokenizer) -> int:
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )

    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()

    if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError("Expected a single tokenized conversation.")
        token_ids = token_ids[0]

    return len(token_ids)


def filter_overlong_examples(df: pd.DataFrame, tokenizer, max_length: int) -> tuple[pd.DataFrame, Counter]:
    if max_length <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}")

    kept_rows = []
    drop_reasons: Counter = Counter()

    for row in df.to_dict(orient="records"):
        sequence_length = compute_message_sequence_length(row["messages"], tokenizer)
        if sequence_length > max_length:
            drop_reasons[f"over_max_length:{max_length}"] += 1
            continue

        kept_row = dict(row)
        kept_row["sequence_length"] = sequence_length
        kept_rows.append(kept_row)

    return pd.DataFrame(kept_rows), drop_reasons


def _truncate_debug_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}... [truncated {len(text) - max_chars} chars]"


def print_debug_samples(df: pd.DataFrame, stage_name: str, num_samples: int, max_chars: int) -> None:
    if num_samples <= 0:
        return
    if df.empty:
        print(f"\n[{stage_name}] No rows available for debug printing.")
        return

    sample_count = min(num_samples, len(df))
    print(f"\n[{stage_name}] Printing {sample_count} sample(s) out of {len(df)} row(s).")

    for sample_idx, row in enumerate(df.head(sample_count).to_dict(orient="records"), start=1):
        print(f"\n[{stage_name}] Sample {sample_idx}")
        print(f"source: {row.get('source', '<missing>')}")
        print(f"assessment: {row.get('assessment', '<missing>')}")
        if "sequence_length" in row:
            print(f"sequence_length: {row['sequence_length']}")

        question = row.get("question")
        if question is not None:
            print("question:")
            print(_truncate_debug_text(str(question), max_chars))

        user_prompt = row.get("user_prompt")
        if user_prompt is not None:
            print("user_prompt:")
            print(_truncate_debug_text(str(user_prompt), max_chars))

        target_text = row.get("target_text")
        if target_text is not None:
            print("target_text:")
            print(_truncate_debug_text(str(target_text), max_chars))

        messages = row.get("messages")
        if messages is not None:
            messages_json = json.dumps(messages, indent=2, ensure_ascii=False)
            print("messages:")
            print(_truncate_debug_text(messages_json, max_chars))

        print("-" * 80)


def _build_stratify_labels(df: pd.DataFrame, columns: Iterable[str]) -> Optional[pd.Series]:
    columns = list(columns)
    if any(column not in df.columns for column in columns):
        return None

    label_df = df[columns].copy()
    for column in columns:
        label_df[column] = label_df[column].map(lambda value: "__missing__" if pd.isna(value) else str(value).strip())

    return label_df.astype(str).agg("||".join, axis=1)


def _is_stratification_feasible(labels: pd.Series, val_ratio: float) -> bool:
    if labels.empty:
        return False

    target_val_size = max(1, int(round(len(labels) * val_ratio)))
    if target_val_size >= len(labels):
        return False

    counts = labels.value_counts(dropna=False)
    if counts.empty or (counts < 2).any():
        return False

    if len(counts) > target_val_size:
        return False

    return True


def _stratified_split_indices(labels: pd.Series, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    target_val_size = max(1, int(round(len(labels) * val_ratio)))
    rng = np.random.default_rng(seed)

    grouped_indices: dict[str, list[int]] = {}
    for idx, label in labels.items():
        grouped_indices.setdefault(str(label), []).append(int(idx))

    allocations: dict[str, int] = {}
    tie_breakers: dict[str, float] = {}
    ranked_labels = []
    base_total = 0

    for label in sorted(grouped_indices):
        shuffled_group = rng.permutation(grouped_indices[label]).tolist()
        grouped_indices[label] = shuffled_group
        ideal = len(shuffled_group) * val_ratio
        base = min(int(np.floor(ideal)), len(shuffled_group) - 1)
        allocations[label] = base
        base_total += base
        tie_breakers[label] = float(rng.random())
        ranked_labels.append((ideal - base, tie_breakers[label], label))

    remaining = target_val_size - base_total
    ranked_labels.sort(key=lambda item: (-item[0], item[1], item[2]))

    while remaining > 0:
        progressed = False
        for _, _, label in ranked_labels:
            if allocations[label] < len(grouped_indices[label]) - 1:
                allocations[label] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break

    val_indices = []
    for label in sorted(grouped_indices):
        val_indices.extend(grouped_indices[label][: allocations[label]])

    val_index_set = set(val_indices)
    train_indices = [idx for idx in labels.index.tolist() if idx not in val_index_set]
    return train_indices, val_indices


def split_processed_dataframe(df: pd.DataFrame, val_ratio: float = DEFAULT_VAL_RATIO, seed: int = DEFAULT_SEED) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if df.empty:
        raise ValueError("Processed dataframe is empty; nothing to split.")
    if len(df) < 2:
        raise ValueError("Need at least 2 valid rows to create train/test splits.")

    stratify_candidates = [
        ("assessment_source", _build_stratify_labels(df, ("assessment", "source"))),
        ("assessment", _build_stratify_labels(df, ("assessment",))),
    ]

    for strategy_name, labels in stratify_candidates:
        if labels is not None and _is_stratification_feasible(labels, val_ratio):
            train_indices, val_indices = _stratified_split_indices(labels=labels, val_ratio=val_ratio, seed=seed)
            train_df = df.loc[train_indices].reset_index(drop=True)
            val_df = df.loc[val_indices].reset_index(drop=True)
            return train_df, val_df, strategy_name

    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(df.index.to_numpy())
    val_size = max(1, int(round(len(df) * val_ratio)))
    val_size = min(val_size, len(df) - 1)
    val_indices = shuffled_indices[:val_size].tolist()
    val_index_set = set(val_indices)
    train_indices = [idx for idx in df.index.tolist() if idx not in val_index_set]

    train_df = df.loc[train_indices].reset_index(drop=True)
    val_df = df.loc[val_indices].reset_index(drop=True)
    return train_df, val_df, "none"


def print_preprocessing_report(
    input_rows: int,
    kept_rows: int,
    drop_reasons: Counter,
    train_size: int,
    test_size: int,
    split_strategy: str,
) -> None:
    print(f"Input rows: {input_rows}")
    print(f"Kept rows: {kept_rows}")
    print(f"Dropped rows: {sum(drop_reasons.values())}")
    if drop_reasons:
        print("Dropped rows by reason:")
        for reason, count in sorted(drop_reasons.items()):
            print(f"  - {reason}: {count}")
    print(f"Split strategy: {split_strategy}")
    print(f"Final train size: {train_size}")
    print(f"Final test size: {test_size}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download PKU-Alignment/self-monitor, convert it to multi-turn SFT parquet, and save train/test splits."
    )
    parser.add_argument("--local_dir", required=True, help="Directory to save train.parquet and test.parquet.")
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS directory to copy the parquet files to.")
    parser.add_argument("--hf_repo_id", default=DEFAULT_HF_REPO_ID, help="Hugging Face dataset repo id.")
    parser.add_argument("--val_ratio", type=float, default=DEFAULT_VAL_RATIO, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for deterministic splitting.")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional tokenizer path/name used to compute chat-template lengths and filter overlong rows.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Optional max sequence length for tokenizer-aware filtering. Must be provided with --tokenizer.",
    )
    parser.add_argument(
        "--debug_print_samples",
        type=int,
        default=0,
        help="Print up to N processed examples after optional length filtering and before train/test split.",
    )
    parser.add_argument(
        "--debug_print_split_samples",
        type=int,
        default=0,
        help="Print up to N examples from each of the final train/test parquet splits.",
    )
    parser.add_argument(
        "--debug_max_chars",
        type=int,
        default=4000,
        help="Maximum characters to print per debug text block before truncation.",
    )
    args = parser.parse_args()

    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {args.val_ratio}")
    if (args.tokenizer is None) != (args.max_length is None):
        raise ValueError("--tokenizer and --max_length must be provided together.")
    if args.max_length is not None and args.max_length <= 0:
        raise ValueError(f"max_length must be positive, got {args.max_length}")
    if args.debug_print_samples < 0:
        raise ValueError(f"debug_print_samples must be non-negative, got {args.debug_print_samples}")
    if args.debug_print_split_samples < 0:
        raise ValueError(f"debug_print_split_samples must be non-negative, got {args.debug_print_split_samples}")
    if args.debug_max_chars <= 0:
        raise ValueError(f"debug_max_chars must be positive, got {args.debug_max_chars}")

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    import datasets

    dataset = datasets.load_dataset(args.hf_repo_id, split="train")
    raw_df = dataset.to_pandas()

    processed_df, drop_reasons = prepare_self_monitor_dataframe(raw_df, system_prompt=SYSTEM_PROMPT)
    if processed_df.empty:
        raise ValueError("No valid rows remain after preprocessing.")

    if args.tokenizer is not None:
        tokenizer = load_chat_tokenizer(args.tokenizer)
        processed_df, length_drop_reasons = filter_overlong_examples(
            processed_df,
            tokenizer=tokenizer,
            max_length=args.max_length,
        )
        drop_reasons.update(length_drop_reasons)
        if processed_df.empty:
            raise ValueError("No valid rows remain after tokenizer-aware length filtering.")

    print_debug_samples(
        processed_df,
        stage_name="processed",
        num_samples=args.debug_print_samples,
        max_chars=args.debug_max_chars,
    )

    train_df, test_df, split_strategy = split_processed_dataframe(processed_df, val_ratio=args.val_ratio, seed=args.seed)

    if args.debug_print_split_samples > 0:
        print_debug_samples(
            train_df,
            stage_name="train",
            num_samples=args.debug_print_split_samples,
            max_chars=args.debug_max_chars,
        )
        print_debug_samples(
            test_df,
            stage_name="test",
            num_samples=args.debug_print_split_samples,
            max_chars=args.debug_max_chars,
        )

    train_path = os.path.join(local_dir, "train.parquet")
    test_path = os.path.join(local_dir, "test.parquet")
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print_preprocessing_report(
        input_rows=len(raw_df),
        kept_rows=len(processed_df),
        drop_reasons=drop_reasons,
        train_size=len(train_df),
        test_size=len(test_df),
        split_strategy=split_strategy,
    )
    print(f"Saved train split to {train_path}")
    print(f"Saved test split to {test_path}")

    if args.hdfs_dir is not None:
        from verl.utils.hdfs_io import copy, makedirs

        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)


if __name__ == "__main__":
    main()