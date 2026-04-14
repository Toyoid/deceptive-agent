import numpy as np

from agent_system.self_monitor import parse_self_monitor_batch, parse_self_monitor_output


def test_parse_self_monitor_output_valid_safe():
    result = parse_self_monitor_output(
        "<think>\nthoughts\n<monitor>checked safety</monitor>\n<label>safe</label>\n</think>\n<answer>ok</answer>"
    )
    assert result.is_valid is True
    assert result.label == "safe"
    assert result.is_unsafe is False
    assert result.invalid_reason == ""


def test_parse_self_monitor_output_valid_unsafe_case_insensitive():
    result = parse_self_monitor_output(
        "<think>\nthoughts\n<monitor>risky content</monitor>\n<label>UNSAFE</label>\n</think>\nfinal"
    )
    assert result.is_valid is True
    assert result.label == "unsafe"
    assert result.is_unsafe is True


def test_parse_self_monitor_output_invalid_cases():
    invalid_cases = [
        ("missing_monitor_block", "<think>\nthoughts\n<label>safe</label>\n</think>\nfinal"),
        ("invalid_label_value", "<think>\nthoughts\n<monitor>ok</monitor>\n<label>maybe</label>\n</think>\nfinal"),
        ("monitor_outside_think", "<think>\nthoughts\n<label>safe</label>\n</think>\n<monitor>oops</monitor>\nfinal"),
        ("invalid_label_tag_count", "<think>\nthoughts\n<monitor>ok</monitor>\n<label>safe</label>\n<label>unsafe</label>\n</think>\nfinal"),
        ("empty_monitor_block", "<think>\nthoughts\n<monitor></monitor>\n<label>safe</label>\n</think>\nfinal"),
        ("empty_think_block", "<think>\n\n</think>\nfinal"),
        ("missing_thinking_process", "<think>\n<monitor>ok</monitor>\n<label>safe</label>\n</think>\nfinal"),
        ("invalid_monitor_label_order", "<think>\nthoughts\n<label>safe</label>\n<monitor>ok</monitor>\n</think>\nfinal"),
        ("invalid_monitor_label_order", "<think>\nthoughts\n<monitor>ok<label>safe</label></monitor>\n</think>\nfinal"),
        ("invalid_monitor_label_order", "<think>\nthoughts\n<monitor>ok</monitor>\nextra\n<label>safe</label>\n</think>\nfinal"),
        ("invalid_monitor_label_order", "<think>\nthoughts\n<monitor>ok</monitor>\n<label>safe</label>\nextra\n</think>\nfinal"),
    ]

    for expected_reason, text in invalid_cases:
        result = parse_self_monitor_output(text)
        assert result.is_valid is False
        assert result.invalid_reason == expected_reason
        assert result.is_unsafe is False


def test_parse_self_monitor_batch_outputs_arrays():
    batch = parse_self_monitor_batch(
        [
            "<think>\na\n<monitor>b</monitor>\n<label>safe</label>\n</think>\nfinal",
            "<think>\na\n<label>unsafe</label>\n</think>\nfinal",
        ]
    )

    np.testing.assert_array_equal(batch["self_monitor_is_valid"], np.array([True, False]))
    np.testing.assert_array_equal(batch["self_monitor_is_unsafe"], np.array([False, False]))
    assert batch["self_monitor_label"].tolist() == ["safe", ""]
    assert batch["self_monitor_invalid_reason"].tolist() == ["", "missing_monitor_block"]
