import asyncio

import numpy as np
import pytest

from agent_system.judge.api_cot_scorer import ApiCotJudgeScorer


def _config(**api_overrides):
    api_cot = {
        "provider": "openai_compatible",
        "model": "dummy-judge",
        "score_regex": r"<score>\s*([0-4])\s*</score>\s*$",
        "parse_error": "raise",
        "api_error": "raise",
    }
    api_cot.update(api_overrides)
    return {
        "valid_tokens": ["0", "1", "2", "3", "4"],
        "token_weights": [-2.0, -1.0, 0.0, 0.5, 1.0],
        "score_profiles": {
            "issue_action": {
                "valid_tokens": ["0", "1", "2", "3", "4"],
                "token_weights": [-2.0, -1.0, 0.0, 0.5, 1.0],
            },
            "no_issue_action": {
                "valid_tokens": ["0", "1"],
                "token_weights": [-1.0, 0.3],
            },
        },
        "api_cot": api_cot,
    }


@pytest.mark.parametrize(
    "token,expected",
    [
        ("0", -2.0),
        ("1", -1.0),
        ("2", 0.0),
        ("3", 0.5),
        ("4", 1.0),
    ],
)
def test_score_texts_accepts_final_score_tags(token, expected):
    scorer = ApiCotJudgeScorer(_config())

    result = scorer.score_texts([f"<think>Checked anchors.</think>\n<score>{token}</score>"], score_profile_name="issue_action")

    assert result.parsed_tokens == [token]
    assert result.scores[0] == pytest.approx(expected)
    expected_probs = np.zeros(5, dtype=np.float32)
    expected_probs[int(token)] = 1.0
    np.testing.assert_allclose(result.token_probs[0], expected_probs)


def test_score_texts_missing_score_maps_to_neutral_error():
    scorer = ApiCotJudgeScorer(_config())

    result = scorer.score_texts(["<think>No final tag.</think>"], score_profile_name="issue_action")

    assert result.parsed_tokens == [None]
    assert result.scores[0] == pytest.approx(0.0)
    assert result.errors[0] is not None


def test_score_texts_out_of_profile_score_maps_to_neutral_error():
    scorer = ApiCotJudgeScorer(_config(score_regex=r"<score>\s*([0-9])\s*</score>\s*$"))

    result = scorer.score_texts(["<think>Bad range.</think>\n<score>5</score>"], score_profile_name="issue_action")

    assert result.parsed_tokens == ["5"]
    assert result.scores[0] == pytest.approx(0.0)
    assert result.errors[0] is not None


def test_score_texts_non_final_score_tag_is_parsed_by_fallback():
    scorer = ApiCotJudgeScorer(_config())

    result = scorer.score_texts(["<score>4</score>\nThis score was not final."], score_profile_name="issue_action")

    assert result.parsed_tokens == ["4"]
    assert result.scores[0] == pytest.approx(1.0)


def test_score_texts_uses_independent_no_issue_profile():
    scorer = ApiCotJudgeScorer(_config())

    result = scorer.score_texts(
        [
            "<think>Correct no issue.</think>\n<score>1</score>",
            "<think>Invalid no-issue label.</think>\n<score>2</score>",
        ],
        score_profile_name="no_issue_action",
    )

    assert result.parsed_tokens == ["1", "2"]
    np.testing.assert_allclose(result.scores, [0.3, 0.0])
    assert result.errors[0] is None
    assert result.errors[1] is not None


def test_score_texts_api_error_raises_by_default():
    scorer = ApiCotJudgeScorer(_config())

    with pytest.raises(RuntimeError, match="API CoT judge call failed"):
        scorer.score_texts([""], errors=["timeout"])


def test_score_texts_api_error_can_be_neutralized_for_rollout():
    scorer = ApiCotJudgeScorer(_config(api_error="neutral"))

    result = scorer.score_texts(
        [""],
        errors=["timeout"],
        score_profile_name="no_issue_action",
    )

    assert result.parsed_tokens == [None]
    assert result.scores[0] == pytest.approx(0.0)
    assert result.errors[0] is not None
    np.testing.assert_allclose(result.token_probs, np.zeros((1, 2), dtype=np.float32))


def test_score_batch_async_generate_batch_error_can_be_neutralized_for_rollout():
    class FailingClient:
        async def generate_batch(self, batch_messages):
            raise RuntimeError("gateway down")

        async def close(self):
            return None

    scorer = ApiCotJudgeScorer(_config(api_error="neutral"))
    scorer._build_client = lambda: FailingClient()

    result = asyncio.run(
        scorer.score_batch_async(
            [[{"role": "user", "content": "judge"}]],
            score_profile_name="issue_action",
        )
    )

    assert result.parsed_tokens == [None]
    assert result.scores[0] == pytest.approx(0.0)
    assert "gateway down" in result.errors[0]
