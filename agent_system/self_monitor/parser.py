from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import numpy as np


_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", flags=re.IGNORECASE | re.DOTALL)
_MONITOR_BLOCK = re.compile(r"<monitor>(.*?)</monitor>", flags=re.IGNORECASE | re.DOTALL)
_LABEL_BLOCK = re.compile(r"<label>(.*?)</label>", flags=re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class SelfMonitorParseResult:
    is_valid: bool
    label: str
    is_unsafe: bool
    invalid_reason: str


def _invalid(reason: str, label: str = "") -> SelfMonitorParseResult:
    return SelfMonitorParseResult(
        is_valid=False,
        label=label,
        is_unsafe=False,
        invalid_reason=reason,
    )


def parse_self_monitor_output(text: str) -> SelfMonitorParseResult:
    raw_text = text if isinstance(text, str) else str(text)

    think_match = _THINK_BLOCK.search(raw_text)
    if think_match is None:
        return _invalid("missing_think_block")

    think_content = think_match.group(1)
    if not think_content.strip():
        return _invalid("empty_think_block")

    monitor_match = _MONITOR_BLOCK.search(think_content)
    if monitor_match is None:
        return _invalid("missing_monitor_block")
    if not monitor_match.group(1).strip():
        return _invalid("empty_monitor_block")

    label_match = _LABEL_BLOCK.search(think_content)
    if label_match is None:
        return _invalid("missing_label_block")

    monitor_start, monitor_end = monitor_match.span()
    label_start, label_end = label_match.span()
    if monitor_start >= label_start or monitor_end > label_start:
        return _invalid("invalid_monitor_label_order")
    if not think_content[:monitor_start].strip():
        return _invalid("missing_thinking_process")
    if think_content[monitor_end:label_start].strip():
        return _invalid("invalid_monitor_label_order")
    if think_content[label_end:].strip():
        return _invalid("invalid_monitor_label_order")

    label = label_match.group(1).strip()
    if not label:
        return _invalid("empty_label_block")

    normalized_label = label.lower()
    if normalized_label not in {"safe", "unsafe"}:
        return _invalid("invalid_label_value", label=label)

    return SelfMonitorParseResult(
        is_valid=True,
        label=normalized_label,
        is_unsafe=normalized_label == "unsafe",
        invalid_reason="",
    )


def parse_self_monitor_batch(texts: Sequence[str]) -> dict[str, np.ndarray]:
    parsed = [parse_self_monitor_output(text) for text in texts]
    return {
        "self_monitor_is_valid": np.asarray([item.is_valid for item in parsed], dtype=bool),
        "self_monitor_label": np.asarray([item.label for item in parsed], dtype=object),
        "self_monitor_is_unsafe": np.asarray([item.is_unsafe for item in parsed], dtype=bool),
        "self_monitor_invalid_reason": np.asarray([item.invalid_reason for item in parsed], dtype=object),
    }
