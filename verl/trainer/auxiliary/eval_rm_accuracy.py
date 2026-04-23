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

"""Evaluate an adopted reward model on PKU-style pairwise safety data.

This script is intentionally separate from auxiliary RL training. The training
side consumes prompt-only data and lets the actor generate online responses,
while RM accuracy evaluation consumes fixed response pairs and checks whether
the RM prefers the annotated safer/better response.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from agent_system.utils.reason_answer_format import extract_visible_answer


PAIR_COLUMNS = {"prompt", "response_0", "response_1"}


def _normalize_chat_prompt(prompt: Any) -> list[dict[str, str]]:
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    if isinstance(prompt, list):
        return [dict(message) for message in prompt]
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    raise TypeError(f"Unsupported prompt type for RM eval: {type(prompt)}")


def _build_rm_text(tokenizer, prompt: Any, response: str, strip_thinking: bool) -> str:
    messages = _normalize_chat_prompt(prompt)
    response_text = extract_visible_answer(response) if strip_thinking else response
    chat = messages + [{"role": "assistant", "content": response_text}]

    if getattr(tokenizer, "chat_template", None) is not None:
        return tokenizer.apply_chat_template(chat, add_generation_prompt=False, tokenize=False)

    # Fallback for tokenizers without chat templates. This is intentionally simple:
    # RM eval should not invent task formatting beyond user text plus assistant text.
    rendered_prompt = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
    return f"{rendered_prompt}\nassistant: {response_text}"


def _torch_dtype(dtype: str):
    import torch

    if dtype == "auto":
        return "auto"
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def load_sequence_rm(model_path: str, tokenizer_path: str | None, trust_remote_code: bool, dtype: str, device: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=_torch_dtype(dtype),
    )
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def score_texts(
    tokenizer,
    model,
    device: str,
    texts: list[str],
    batch_size: int,
    max_length: int | None,
    score_index: int,
    desc: str,
) -> np.ndarray:
    import torch

    scores: list[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch_texts = texts[start : start + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
            add_special_tokens=False,
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits

        if logits.ndim == 1:
            batch_scores = logits
        elif logits.shape[-1] == 1:
            batch_scores = logits.squeeze(-1)
        else:
            batch_scores = logits[:, score_index]
        scores.append(batch_scores.detach().float().cpu().numpy())

    return np.concatenate(scores, axis=0)


def _pairwise_correct(score_0: np.ndarray, score_1: np.ndarray, label: np.ndarray, tie_policy: str) -> np.ndarray:
    pred = np.where(score_1 > score_0, 1, 0)
    ties = score_0 == score_1
    correct = (pred == label).astype(np.float32)
    if tie_policy == "half":
        correct[ties] = 0.5
    elif tie_policy == "incorrect":
        correct[ties] = 0.0
    else:
        raise ValueError(f"Unsupported tie_policy: {tie_policy}")
    return correct


def _metric(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(values.mean())


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    df = pd.read_parquet(args.data_path)
    if args.max_samples is not None and args.max_samples > 0:
        df = df.head(args.max_samples)

    missing = PAIR_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Data parquet is missing required columns: {sorted(missing)}")

    tokenizer, model, device = load_sequence_rm(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        device=args.device,
    )

    texts_0 = [
        _build_rm_text(tokenizer, prompt, response, args.strip_thinking)
        for prompt, response in zip(df["prompt"], df["response_0"])
    ]
    texts_1 = [
        _build_rm_text(tokenizer, prompt, response, args.strip_thinking)
        for prompt, response in zip(df["prompt"], df["response_1"])
    ]

    score_0 = score_texts(tokenizer, model, device, texts_0, args.batch_size, args.max_length, args.score_index, "Scoring response_0")
    score_1 = score_texts(tokenizer, model, device, texts_1, args.batch_size, args.max_length, args.score_index, "Scoring response_1")

    result: dict[str, Any] = {
        "data_path": args.data_path,
        "model": args.model,
        "num_examples": int(len(df)),
        "score_0_mean": float(score_0.mean()) if len(score_0) else None,
        "score_1_mean": float(score_1.mean()) if len(score_1) else None,
    }

    if "safer_response_id" in df.columns:
        safer_label = df["safer_response_id"].astype(int).to_numpy()
        safer_correct = _pairwise_correct(score_0, score_1, safer_label, args.tie_policy)
        result["safer_accuracy"] = _metric(safer_correct)

    if "better_response_id" in df.columns:
        better_label = df["better_response_id"].astype(int).to_numpy()
        better_correct = _pairwise_correct(score_0, score_1, better_label, args.tie_policy)
        result["better_accuracy"] = _metric(better_correct)

    if args.output_json is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a sequence-classification RM on PKU-style pairs.")
    parser.add_argument("--data_path", required=True, help="Pairwise parquet produced by auxiliary preprocess pku_rm_eval.")
    parser.add_argument("--model", required=True, help="Reward model path or HuggingFace id.")
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer path. Defaults to --model.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--score_index", type=int, default=-1, help="Logit index for multi-logit classifiers.")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--strip_thinking", action="store_true")
    parser.add_argument("--tie_policy", choices=["half", "incorrect"], default="half")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def main() -> None:
    result = evaluate(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
