import numpy as np
import pytest

from agent_system.verdict_monitor import constrained_probs_to_binary_penalties


def test_constrained_probs_to_binary_penalties_uses_argmax_token_mapping():
    probs = np.asarray(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.51, 0.49],
        ],
        dtype=np.float32,
    )

    penalties = constrained_probs_to_binary_penalties(probs, valid_tokens=["0", "1"])

    np.testing.assert_array_equal(penalties, np.asarray([0.0, 1.0, 0.0], dtype=np.float32))


def test_constrained_probs_to_binary_penalties_respects_token_order():
    probs = np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32)

    penalties = constrained_probs_to_binary_penalties(probs, valid_tokens=["1", "0"])

    np.testing.assert_array_equal(penalties, np.asarray([1.0, 0.0], dtype=np.float32))


def test_constrained_probs_to_binary_penalties_requires_binary_tokens():
    with pytest.raises(ValueError, match="valid_tokens must be exactly"):
        constrained_probs_to_binary_penalties(np.asarray([[1.0]], dtype=np.float32), valid_tokens=["unsafe"])


def test_constrained_probs_to_binary_penalties_rejects_extra_labels():
    with pytest.raises(ValueError, match="valid_tokens must be exactly"):
        constrained_probs_to_binary_penalties(
            np.asarray([[0.1, 0.2, 0.7]], dtype=np.float32),
            valid_tokens=["0", "1", "2"],
        )
