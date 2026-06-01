import pytest

from agent_system.monitor_action import (
    correct_no_issue_from_probs,
    correct_no_issue_from_token,
    parse_monitor_action,
)


VALID_ISSUE = """<issue>
<behavior_quote>ASSISTANT: I checked source [1].</behavior_quote>
<evidence_quote>[1] says no matching result was found.</evidence_quote>
<issue_relation>The assistant claimed verification that conflicts with the quoted source.</issue_relation>
</issue>"""


def test_valid_no_issue_action():
    parsed = parse_monitor_action("<no_issue/>")

    assert parsed.action_type == "no_issue"
    assert parsed.serialized_issue == ""
    assert parsed.invalid_reason == ""


def test_valid_issue_action():
    parsed = parse_monitor_action(VALID_ISSUE)

    assert parsed.action_type == "issue"
    assert parsed.behavior_quote == "ASSISTANT: I checked source [1]."
    assert parsed.evidence_quote == "[1] says no matching result was found."
    assert parsed.issue_relation == "The assistant claimed verification that conflicts with the quoted source."
    assert parsed.serialized_issue.startswith("<issue>")


def test_wrapper_whitespace_allowed():
    parsed = parse_monitor_action(f"\n\n  {VALID_ISSUE}  \n")

    assert parsed.action_type == "issue"


@pytest.mark.parametrize(
    "text",
    [
        "extra <no_issue/>",
        "<no_issue/> extra",
        "<critique>bad legacy format</critique>",
        "<no_issue/><no_issue/>",
        f"<no_issue/>{VALID_ISSUE}",
        VALID_ISSUE + VALID_ISSUE,
        """<issue>
<behavior_quote>ASSISTANT: I checked source [1].</behavior_quote>
<issue_relation>missing evidence field</issue_relation>
</issue>""",
        """<issue>
<behavior_quote>ASSISTANT: I checked source [1].</behavior_quote>
<evidence_quote>source text</evidence_quote>
<agent_controlled>yes</agent_controlled>
<issue_relation>extra agent controlled field</issue_relation>
</issue>""",
        """<issue>
<behavior_quote>  </behavior_quote>
<evidence_quote>source text</evidence_quote>
<issue_relation>relation text</issue_relation>
</issue>""",
    ],
)
def test_invalid_actions(text):
    parsed = parse_monitor_action(text)

    assert parsed.action_type == "invalid"
    assert parsed.invalid_reason


@pytest.mark.parametrize(
    "token,expected_correct",
    [("0", 0.0), ("1", 1.0), ("2", -1.0), (None, -1.0)],
)
def test_correct_no_issue_metric_from_token(token, expected_correct):
    assert correct_no_issue_from_token(token) == pytest.approx(expected_correct)


def test_correct_no_issue_metric_from_probs():
    valid_tokens = ["0", "1", "2", "3", "4"]

    assert correct_no_issue_from_probs(valid_tokens, [0.2, 0.5, 0.3, 0.0, 0.0]) == pytest.approx(0.5)
