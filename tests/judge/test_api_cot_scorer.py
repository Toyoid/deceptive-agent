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
        "token_weights": [-1.0, -0.5, 0.0, 0.5, 1.0],
        "api_cot": api_cot,
    }


@pytest.mark.parametrize(
    "token,expected",
    [
        ("0", -1.0),
        ("1", -0.5),
        ("2", 0.0),
        ("3", 0.5),
        ("4", 1.0),
    ],
)
def test_score_texts_accepts_final_score_tags(token, expected):
    scorer = ApiCotJudgeScorer(_config())

    result = scorer.score_texts([f"<think>Checked anchors.</think>\n<score>{token}</score>"])

    assert result.parsed_tokens == [token]
    assert result.scores[0] == pytest.approx(expected)
    expected_probs = np.zeros(5, dtype=np.float32)
    expected_probs[int(token)] = 1.0
    np.testing.assert_allclose(result.token_probs[0], expected_probs)


def test_score_texts_rejects_missing_score():
    scorer = ApiCotJudgeScorer(_config())

    with pytest.raises(ValueError, match="no final score tag"):
        scorer.score_texts(["<think>No final tag.</think>"])


def test_score_texts_rejects_out_of_range_score():
    scorer = ApiCotJudgeScorer(_config(score_regex=r"<score>\s*([0-9])\s*</score>\s*$"))

    with pytest.raises(ValueError, match="not in valid_tokens"):
        scorer.score_texts(["<think>Bad range.</think>\n<score>5</score>"])


def test_score_texts_rejects_non_final_score_tag():
    scorer = ApiCotJudgeScorer(_config())

    with pytest.raises(ValueError, match="no final score tag"):
        scorer.score_texts(["<score>4</score>\nThis score was not final."])


def test_score_texts_api_error_raises_by_default():
    scorer = ApiCotJudgeScorer(_config())

    with pytest.raises(RuntimeError, match="API CoT judge call failed"):
        scorer.score_texts([""], errors=["timeout"])
