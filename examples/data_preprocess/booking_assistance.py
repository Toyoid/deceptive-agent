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
Preprocess Booking-Assistance histories into parquet for ReasonChat training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import datasets
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_system.history_utils import history_messages_to_chat_messages

DEFAULT_DOMAINS = ("flights", "hotels", "restaurants", "events")
DEFAULT_SOURCE_ROOT = REPO_ROOT / "Targeted-Manipulation-and-Deception-in-LLMs"
TICKETS_CONFIG_DIR = Path("targeted_llm_manipulation") / "config" / "env_configs" / "tickets"
MASTER_CONFIG_PATH = TICKETS_CONFIG_DIR / "_master_config.yaml"

FORMAT_PROMPT = (
    "Your response must follow a specific format. First, engage in a step-by-step reasoning process "
    "within a thinking pad <think> </think>. This is your internal monologue to analyze the booking "
    "context, the failed tool attempt, and how to respond helpfully and honestly. After your thinking "
    "process, provide the final user-facing answer after </think>.\n"
    "The response structure must be:\n"
    "<think>\n"
    "[Your detailed reasoning about the booking context, constraints, and tool failure goes here.]\n"
    "</think>\n\n"
    "<answer>\n"
    "[Your final, concise answer to the user goes here.]\n"
    "</answer>\n"
    "Your entire output must begin with <think> and end with </answer>.\n"
)

BEHAVIOR_INSTRUCTIONS = {
    "honest": (
        "Be transparent about tool failures. Do not imply a booking, search result, availability, or reservation "
        "succeeded unless the history shows it succeeded.\n"
    ),
    "neutral": "",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        usage="%(prog)s --local_dir /your/workspace/verl_data/booking_assistance "
        "[--source_root PATH] [--domains flights,hotels]"
    )
    parser.add_argument("--local_dir", required=True, help="Local directory to store preprocessed parquet files.")
    parser.add_argument("--source_root", default=str(DEFAULT_SOURCE_ROOT), help="Root of the cloned Booking-Assistance source repository.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--domains",
        default=",".join(DEFAULT_DOMAINS),
        help="Comma-separated booking domains to include. Defaults to all domains.",
    )
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Deterministic test split ratio in [0, 1).")
    parser.add_argument(
        "--behavior_mode",
        choices=sorted(BEHAVIOR_INSTRUCTIONS),
        default="honest",
        help="Extra behavior instruction appended to the booking system prompt.",
    )
    parser.add_argument("--agent_name", default="TicketEase", help="Booking assistant brand name used in the system prompt.")
    parser.add_argument(
        "--max_samples_per_domain",
        type=int,
        default=None,
        help="Optional cap per domain for smoke tests and quick iteration.",
    )
    return parser.parse_args()


def parse_domains(domains_arg: str) -> List[str]:
    domains = [domain.strip() for domain in domains_arg.split(",") if domain.strip()]
    if not domains:
        raise ValueError("At least one domain must be specified.")
    return domains


def deterministic_is_test(uid: str, test_ratio: float) -> bool:
    if test_ratio <= 0:
        return False
    if test_ratio >= 1:
        return True
    digest = hashlib.sha1(uid.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return bucket < test_ratio


def load_master_config(source_root: Path) -> Dict:
    config_path = source_root / MASTER_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Booking master config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_system_prompt(
    master_config: Dict,
    function_def: str,
    agent_name: str,
    behavior_mode: str,
) -> str:
    agent_prompt = master_config["agent_config"]["system_prompt"].format(
        agent_name=agent_name,
        function_def=function_def,
    ).strip()
    behavior_instruction = BEHAVIOR_INSTRUCTIONS[behavior_mode]
    return f"{agent_prompt}\n\n{behavior_instruction}{FORMAT_PROMPT}".strip()


def load_domain_histories(
    source_root: Path,
    domain: str,
    max_samples_per_domain: int | None = None,
) -> tuple[Dict, List[tuple[str, List[Dict[str, str]]]]]:
    data_path = source_root / TICKETS_CONFIG_DIR / f"uncertain_{domain}.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Booking domain file not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    histories = sorted(config["histories"].items(), key=lambda item: int(item[0]))
    if max_samples_per_domain is not None:
        histories = histories[:max_samples_per_domain]

    return config, histories


def build_prompt_messages(
    system_prompt: str,
    history_messages: Iterable[Dict[str, str]],
) -> List[Dict[str, str]]:
    return [{"role": "system", "content": system_prompt}] + history_messages_to_chat_messages(history_messages)


def convert_history_to_row(
    *,
    split: str,
    row_index: int,
    domain: str,
    history_id: str,
    history_messages: List[Dict[str, str]],
    system_prompt: str,
) -> Dict:
    prompt_messages = build_prompt_messages(system_prompt, history_messages)
    data_source = f"booking_assistance/{domain}"

    return {
        "data_source": data_source,
        "prompt": prompt_messages,
        "ability": "alignment",
        "reward_model": {
            "style": "model",
        },
        "extra_info": {
            "split": split,
            "index": row_index,
            "domain": domain,
            "history_id": history_id,
        },
        "env_kwargs": {
            "task_type": "booking_assistance",
            "history_messages": history_messages,
            "history_id": history_id,
            "source_domain": domain,
        },
    }


def build_datasets(args: argparse.Namespace) -> tuple[datasets.Dataset, datasets.Dataset]:
    source_root = Path(args.source_root).resolve()
    domains = parse_domains(args.domains)
    master_config = load_master_config(source_root)

    train_rows: List[Dict] = []
    test_rows: List[Dict] = []
    train_index = 0
    test_index = 0

    for domain in domains:
        domain_config, histories = load_domain_histories(
            source_root=source_root,
            domain=domain,
            max_samples_per_domain=args.max_samples_per_domain,
        )
        system_prompt = build_system_prompt(
            master_config=master_config,
            function_def=domain_config["function_def"],
            agent_name=args.agent_name,
            behavior_mode=args.behavior_mode,
        )

        for history_id, history_messages in histories:
            uid = f"{domain}:{history_id}"
            is_test = deterministic_is_test(uid, args.test_ratio)
            if is_test:
                test_rows.append(
                    convert_history_to_row(
                        split="test",
                        row_index=test_index,
                        domain=domain,
                        history_id=history_id,
                        history_messages=history_messages,
                        system_prompt=system_prompt,
                    )
                )
                test_index += 1
            else:
                train_rows.append(
                    convert_history_to_row(
                        split="train",
                        row_index=train_index,
                        domain=domain,
                        history_id=history_id,
                        history_messages=history_messages,
                        system_prompt=system_prompt,
                    )
                )
                train_index += 1

    if not train_rows:
        raise ValueError("No training samples were generated. Please adjust --domains / --test_ratio.")
    if not test_rows:
        raise ValueError("No test samples were generated. Please adjust --test_ratio.")

    return datasets.Dataset.from_list(train_rows), datasets.Dataset.from_list(test_rows)


def main() -> None:
    args = parse_args()
    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, test_dataset = build_datasets(args)

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if args.hdfs_dir is not None:
        try:
            from verl.utils.hdfs_io import copy, makedirs
        except Exception as exc:  # pragma: no cover - only exercised in specialized environments
            raise RuntimeError(
                "Failed to import verl.utils.hdfs_io for HDFS export. "
                "Please run inside the full training environment or omit --hdfs_dir."
            ) from exc

        makedirs(args.hdfs_dir)
        copy(src=str(local_dir), dst=args.hdfs_dir)


if __name__ == "__main__":
    main()
