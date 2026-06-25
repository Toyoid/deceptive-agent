import pytest

from agent_system.grm.reward import GrmRewardConfig, score_grm_output
from agent_system.judge.score_parsing import parse_score_token


@pytest.mark.parametrize(
    "text,expected",
    [
        ("<think>checked</think>\n<score>4</score>", "4"),
        ("<think>old</think> notes </think>\n<score>3</score>", "3"),
        ("2", "2"),
        ("final: 1", "1"),
        ("score = 0", "0"),
    ],
)
def test_parse_score_token_shared_patterns(text, expected):
    assert parse_score_token(text) == expected


def test_parse_score_token_rejects_missing_score():
    with pytest.raises(ValueError):
        parse_score_token("<think>No final score here.</think>")


def test_grm_reward_exact_label():
    result = score_grm_output(
        "<think>valid</think>\n<score>4</score>",
        label="4",
        valid_tokens=["0", "1", "2", "3", "4"],
    )

    assert result.reward == pytest.approx(1.0)
    assert result.parsed_token == "4"
    assert result.exact


def test_grm_reward_same_direction_adjacent_issue_action():
    result = score_grm_output(
        "<score>3</score>",
        label="4",
        valid_tokens=["0", "1", "2", "3", "4"],
    )

    assert result.reward == pytest.approx(0.0)
    assert result.adjacent_same_direction


def test_grm_reward_wrong_valid_token():
    result = score_grm_output(
        "<score>2</score>",
        label="4",
        valid_tokens=["0", "1", "2", "3", "4"],
    )

    assert result.reward == pytest.approx(-1.0)
    assert not result.adjacent_same_direction


def test_grm_reward_invalid_format_and_wrong_profile_token_are_small_penalty():
    cfg = GrmRewardConfig(invalid_reward=-0.1)

    missing = score_grm_output(
        "no final score",
        label="1",
        valid_tokens=["0", "1"],
        config=cfg,
    )
    wrong_profile = score_grm_output(
        "<score>4</score>",
        label="1",
        valid_tokens=["0", "1"],
        config=cfg,
    )

    assert missing.reward == pytest.approx(-0.1)
    assert missing.invalid_format
    assert wrong_profile.reward == pytest.approx(-0.1)
    assert wrong_profile.wrong_profile_token
