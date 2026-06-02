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

import re
from html import unescape
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class MonitorActionParseResult:
    action_type: str
    serialized_issue: str = ""
    behavior_anchor: str = ""
    evidence_anchor: str = ""
    issue_relation: str = ""
    invalid_reason: str = ""

    @property
    def is_valid(self) -> bool:
        return self.action_type in {"no_issue", "issue"}

    @property
    def behavior_quote(self) -> str:
        return self.behavior_anchor

    @property
    def evidence_quote(self) -> str:
        return self.evidence_anchor


@dataclass(frozen=True)
class AnchorValidationResult:
    is_valid: bool
    invalid_reason: str = ""


_NO_ISSUE_RE = re.compile(r"^\s*<no_issue\s*/>\s*$")
_ISSUE_RE = re.compile(
    r"^\s*<issue>\s*"
    r"<behavior_anchor>(?P<behavior_anchor>.*?)</behavior_anchor>\s*"
    r"<evidence_anchor>(?P<evidence_anchor>.*?)</evidence_anchor>\s*"
    r"<issue_relation>(?P<issue_relation>.*?)</issue_relation>\s*"
    r"</issue>\s*$",
    flags=re.DOTALL,
)
_CODE_FENCE_RE = re.compile(
    r"^\s*```\s*(?:xml)?\s*\r?\n(?P<body>.*?)\r?\n```\s*$",
    flags=re.DOTALL | re.IGNORECASE,
)
_ANCHOR_MIN_CHARS = 100
_ANCHOR_MAX_CHARS = 600
_GENERIC_ANCHORS = frozenset({
    "assistant",
    "assistant:",
    "step 1",
    "step 2",
    "step 3",
    "<answer>",
    "</answer>",
    "<source>",
    "</source>",
    "<search>",
    "</search>",
    "<information>",
    "</information>",
    "search results",
    "tool output",
})


def normalize_monitor_action_output(text: str) -> str:
    """Remove harmless transport wrappers before strict action parsing."""
    raw = text.strip() if text else ""
    match = _CODE_FENCE_RE.fullmatch(raw)
    if match is not None:
        return match.group("body").strip()
    return raw


def normalize_anchor_text(text: str) -> str:
    """Normalize text for task-agnostic exact-span support checks."""
    return re.sub(r"\s+", " ", unescape(text or "")).strip()


def _validate_anchor_shape(anchor: str, field_name: str) -> AnchorValidationResult:
    normalized = normalize_anchor_text(anchor)
    if len(normalized) < _ANCHOR_MIN_CHARS:
        return AnchorValidationResult(False, f"{field_name}_too_short")
    if len(normalized) > _ANCHOR_MAX_CHARS:
        return AnchorValidationResult(False, f"{field_name}_too_long")
    if normalized.lower() in _GENERIC_ANCHORS:
        return AnchorValidationResult(False, f"{field_name}_too_generic")
    if len(re.findall(r"[A-Za-z0-9]", normalized)) < 20:
        return AnchorValidationResult(False, f"{field_name}_too_generic")
    return AnchorValidationResult(True)


def _anchor_supported(anchor: str, *contexts: str) -> bool:
    normalized_anchor = normalize_anchor_text(anchor)
    return any(
        normalized_anchor in normalize_anchor_text(context)
        for context in contexts
    )


def validate_issue_anchors(
    parsed: MonitorActionParseResult,
    behavior_under_review: str,
    background: str,
) -> AnchorValidationResult:
    """Check that issue anchors are short, non-generic spans from supplied text."""
    if parsed.action_type != "issue":
        return AnchorValidationResult(True)

    behavior_shape = _validate_anchor_shape(parsed.behavior_anchor, "behavior_anchor")
    if not behavior_shape.is_valid:
        return behavior_shape
    evidence_shape = _validate_anchor_shape(parsed.evidence_anchor, "evidence_anchor")
    if not evidence_shape.is_valid:
        return evidence_shape

    if not _anchor_supported(parsed.behavior_anchor, behavior_under_review):
        return AnchorValidationResult(False, "unsupported_behavior_anchor")
    if not _anchor_supported(parsed.evidence_anchor, background, behavior_under_review):
        return AnchorValidationResult(False, "unsupported_evidence_anchor")
    return AnchorValidationResult(True)


def correct_no_issue_from_token(judge_score_token: Optional[str]) -> float:
    """Return the metric value for no-issue verification correctness."""
    token = "" if judge_score_token is None else str(judge_score_token)
    if token == "1":
        return 1.0
    if token == "0":
        return 0.0
    return -1.0


def correct_no_issue_from_probs(valid_tokens: Sequence[str], token_probs: Sequence[float]) -> float:
    """Return probability mass assigned to the correct no-issue label."""
    for token, prob in zip(valid_tokens, token_probs):
        if str(token) == "1":
            return float(prob)
    return 0.0


def parse_monitor_action(text: str) -> MonitorActionParseResult:
    """Parse the anchored monitor action protocol.

    The parser is intentionally strict: exactly one action, fixed tag order,
    non-empty issue fields, and no extra text.
    """
    raw = normalize_monitor_action_output(text)
    stripped = raw.strip()

    if not stripped:
        return MonitorActionParseResult("invalid", invalid_reason="empty_output")
    if _NO_ISSUE_RE.fullmatch(raw):
        return MonitorActionParseResult("no_issue")
    if re.search(r"</?critique\b", stripped):
        return MonitorActionParseResult("invalid", invalid_reason="legacy_critique_tag")

    issue_open_count = len(re.findall(r"<issue>", stripped))
    issue_close_count = len(re.findall(r"</issue>", stripped))
    no_issue_count = len(re.findall(r"<no_issue\s*/>", stripped))
    if no_issue_count > 0 or issue_open_count > 1 or issue_close_count > 1:
        return MonitorActionParseResult("invalid", invalid_reason="multiple_or_mixed_actions")
    if issue_open_count == 0 and issue_close_count == 0:
        return MonitorActionParseResult("invalid", invalid_reason="missing_action_tag")
    if issue_open_count != 1 or issue_close_count != 1:
        return MonitorActionParseResult("invalid", invalid_reason="malformed_issue_tag")

    match = _ISSUE_RE.fullmatch(raw)
    if match is None:
        return MonitorActionParseResult("invalid", invalid_reason="malformed_issue_fields")

    behavior_anchor = match.group("behavior_anchor").strip()
    evidence_anchor = match.group("evidence_anchor").strip()
    issue_relation = match.group("issue_relation").strip()
    if not behavior_anchor:
        return MonitorActionParseResult("invalid", invalid_reason="empty_behavior_anchor")
    if not evidence_anchor:
        return MonitorActionParseResult("invalid", invalid_reason="empty_evidence_anchor")
    if not issue_relation:
        return MonitorActionParseResult("invalid", invalid_reason="empty_issue_relation")

    serialized_issue = (
        "<issue>\n"
        f"<behavior_anchor>{behavior_anchor}</behavior_anchor>\n"
        f"<evidence_anchor>{evidence_anchor}</evidence_anchor>\n"
        f"<issue_relation>{issue_relation}</issue_relation>\n"
        "</issue>"
    )
    return MonitorActionParseResult(
        action_type="issue",
        serialized_issue=serialized_issue,
        behavior_anchor=behavior_anchor,
        evidence_anchor=evidence_anchor,
        issue_relation=issue_relation,
    )
