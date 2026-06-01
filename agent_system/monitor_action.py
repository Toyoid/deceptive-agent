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
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class MonitorActionParseResult:
    action_type: str
    serialized_issue: str = ""
    behavior_quote: str = ""
    evidence_quote: str = ""
    issue_relation: str = ""
    invalid_reason: str = ""

    @property
    def is_valid(self) -> bool:
        return self.action_type in {"no_issue", "issue"}


_NO_ISSUE_RE = re.compile(r"^\s*<no_issue\s*/>\s*$")
_ISSUE_RE = re.compile(
    r"^\s*<issue>\s*"
    r"<behavior_quote>(?P<behavior_quote>.*?)</behavior_quote>\s*"
    r"<evidence_quote>(?P<evidence_quote>.*?)</evidence_quote>\s*"
    r"<issue_relation>(?P<issue_relation>.*?)</issue_relation>\s*"
    r"</issue>\s*$",
    flags=re.DOTALL,
)


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
    raw = text or ""
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

    behavior_quote = match.group("behavior_quote").strip()
    evidence_quote = match.group("evidence_quote").strip()
    issue_relation = match.group("issue_relation").strip()
    if not behavior_quote:
        return MonitorActionParseResult("invalid", invalid_reason="empty_behavior_quote")
    if not evidence_quote:
        return MonitorActionParseResult("invalid", invalid_reason="empty_evidence_quote")
    if not issue_relation:
        return MonitorActionParseResult("invalid", invalid_reason="empty_issue_relation")

    serialized_issue = (
        "<issue>\n"
        f"<behavior_quote>{behavior_quote}</behavior_quote>\n"
        f"<evidence_quote>{evidence_quote}</evidence_quote>\n"
        f"<issue_relation>{issue_relation}</issue_relation>\n"
        "</issue>"
    )
    return MonitorActionParseResult(
        action_type="issue",
        serialized_issue=serialized_issue,
        behavior_quote=behavior_quote,
        evidence_quote=evidence_quote,
        issue_relation=issue_relation,
    )
