"""Shared parsing for CoT judge final score tokens."""

from __future__ import annotations

import re
from typing import Optional


DEFAULT_SCORE_REGEX = r"<score>\s*([0-4])\s*</score>\s*$"


def parse_score_token(raw_output: str, score_regex: Optional[str] = None) -> str:
    """Extract the final judge score token from a generated judge response."""
    text = raw_output or ""
    configured_regex = score_regex or DEFAULT_SCORE_REGEX

    # 1) Preferred: user-configured regex (usually <score>...</score> at end).
    match = re.search(configured_regex, text, flags=re.DOTALL)
    if match is not None:
        return match.group(1).strip()

    # 2) Strip <think>...</think> blocks and anything before the last </think>.
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if re.search(r"</think>", stripped, flags=re.IGNORECASE):
        stripped = re.split(r"</think>", stripped, flags=re.IGNORECASE)[-1]
    stripped = stripped.strip()

    # 2a) Accept <score> N </score> anywhere in the stripped content.
    match = re.search(r"<score>\s*([0-9]+)\s*</score>", stripped, flags=re.DOTALL | re.IGNORECASE)
    if match is not None:
        return match.group(1).strip()

    # 2b) Accept a single-token output (exact match).
    if re.fullmatch(r"[0-9]+", stripped):
        return stripped

    # 2c) Accept labeled patterns like "score: N" or "final: N".
    label_pattern = r"(?:score|final|answer|verdict|result|output)\s*[:=]\s*([0-9]+)"
    matches = re.findall(label_pattern, stripped, flags=re.IGNORECASE)
    if matches:
        return matches[-1].strip()

    raise ValueError(
        "no parseable score token found; expected a score token "
        f"matching score_regex={configured_regex!r}; output={raw_output!r}"
    )
