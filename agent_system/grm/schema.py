"""Schema helpers for GRM judge-sample JSONL rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


GRM_SAMPLE_FIELDS = ("prompt", "valid_tokens", "judge_pred_token", "label")


def normalize_valid_tokens(valid_tokens: Any) -> List[str]:
    if valid_tokens is None:
        return []
    return [str(token) for token in list(valid_tokens)]


def normalize_prompt(prompt: Any) -> List[Dict[str, str]]:
    if prompt is None:
        return []
    messages = prompt.tolist() if hasattr(prompt, "tolist") else prompt
    normalized = []
    for message in list(messages):
        normalized.append({
            "role": str(message.get("role", "")),
            "content": str(message.get("content", "")),
        })
    return normalized


@dataclass(frozen=True)
class GrmJudgeSample:
    prompt: List[Dict[str, str]]
    valid_tokens: List[str]
    judge_pred_token: Optional[str] = None
    label: Optional[str] = None

    @classmethod
    def from_dict(cls, row: Dict[str, Any]) -> "GrmJudgeSample":
        pred = row.get("judge_pred_token")
        label = row.get("label")
        return cls(
            prompt=normalize_prompt(row.get("prompt")),
            valid_tokens=normalize_valid_tokens(row.get("valid_tokens")),
            judge_pred_token=None if pred is None else str(pred),
            label=None if label is None or label == "" else str(label),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "valid_tokens": list(self.valid_tokens),
            "judge_pred_token": self.judge_pred_token,
            "label": self.label,
        }

    def validate_for_training(self) -> None:
        if not self.prompt:
            raise ValueError("GRM sample missing prompt")
        if not self.valid_tokens:
            raise ValueError("GRM sample missing valid_tokens")
        if self.label is None:
            raise ValueError("GRM sample missing label")
        if self.label not in self.valid_tokens:
            raise ValueError(
                f"GRM sample label {self.label!r} is not in valid_tokens={self.valid_tokens!r}"
            )
