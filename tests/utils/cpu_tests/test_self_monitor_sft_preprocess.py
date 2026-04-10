import importlib.util
from pathlib import Path

import pandas as pd


_MODULE_PATH = Path(__file__).resolve().parents[3] / "examples" / "data_preprocess" / "self_monitor_sft.py"
_SPEC = importlib.util.spec_from_file_location("self_monitor_sft_for_test", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)

SYSTEM_PROMPT = _MODULE.SYSTEM_PROMPT
filter_overlong_examples = _MODULE.filter_overlong_examples
prepare_self_monitor_dataframe = _MODULE.prepare_self_monitor_dataframe
split_processed_dataframe = _MODULE.split_processed_dataframe


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        assert tokenize is True
        assert add_generation_prompt is False
        serialized = "\n".join(f"{message['role']}:{message['content']}" for message in messages)
        return list(range(len(serialized)))


def test_prepare_self_monitor_dataframe_formats_and_drops_invalid_rows():
    raw_df = pd.DataFrame(
        [
            {
                "question": "  How should I respond?  ",
                "thinking_process": "  I should reason carefully.  ",
                "reflection": "  Check for safety risks. ",
                "assessment": " safe ",
                "revised_response": " Give a careful answer. ",
                "source": "model-a",
            },
            {
                "question": " ",
                "thinking_process": "Reasoning",
                "reflection": "Reflection",
                "assessment": "safe",
                "revised_response": "Answer",
                "source": "model-b",
            },
            {
                "question": "Bad row",
                "thinking_process": "Reasoning",
                "reflection": "Contains </monitor> reserved tag",
                "assessment": "unsafe",
                "revised_response": "Answer",
                "source": "model-c",
            },
        ]
    )

    processed_df, drop_reasons = prepare_self_monitor_dataframe(raw_df, system_prompt=SYSTEM_PROMPT)

    assert len(processed_df) == 1
    assert drop_reasons["missing_or_empty:question"] == 1
    assert drop_reasons["reserved_tag:</monitor>"] == 1

    row = processed_df.iloc[0]
    expected_target = (
        "<think>\n"
        "I should reason carefully.\n"
        "<monitor>Check for safety risks.</monitor>\n"
        "<label>safe</label>\n"
        "</think>\n"
        "Give a careful answer."
    )
    expected_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "How should I respond?"},
        {"role": "assistant", "content": expected_target},
    ]

    assert row["question"] == "How should I respond?"
    assert row["thinking_process"] == "I should reason carefully."
    assert row["reflection"] == "Check for safety risks."
    assert row["assessment"] == "safe"
    assert row["revised_response"] == "Give a careful answer."
    assert row["system_prompt"] == SYSTEM_PROMPT
    assert row["target_text"] == expected_target
    assert row["messages"] == expected_messages


def test_filter_overlong_examples_drops_rows_before_training():
    raw_df = pd.DataFrame(
        [
            {
                "question": "Short question",
                "thinking_process": "Short reasoning",
                "reflection": "Short reflection",
                "assessment": "safe",
                "revised_response": "Short answer",
                "source": "model-a",
            },
            {
                "question": "Long question " * 30,
                "thinking_process": "Long reasoning " * 80,
                "reflection": "Long reflection " * 40,
                "assessment": "unsafe",
                "revised_response": "Long answer " * 40,
                "source": "model-b",
            },
        ]
    )

    processed_df, drop_reasons = prepare_self_monitor_dataframe(raw_df, system_prompt=SYSTEM_PROMPT)

    assert not drop_reasons

    filtered_df, length_drop_reasons = filter_overlong_examples(
        processed_df,
        tokenizer=_FakeTokenizer(),
        max_length=400,
    )

    assert len(filtered_df) == 1
    assert length_drop_reasons["over_max_length:400"] == 1
    assert filtered_df.iloc[0]["question"] == "Short question"
    assert isinstance(filtered_df.iloc[0]["sequence_length"], int)
    assert filtered_df.iloc[0]["sequence_length"] <= 400


def test_split_processed_dataframe_is_deterministic_and_falls_back_to_assessment():
    raw_rows = []
    for idx in range(10):
        raw_rows.append(
            {
                "question": f"safe question {idx}",
                "thinking_process": f"safe reasoning {idx}",
                "reflection": f"safe reflection {idx}",
                "assessment": "safe",
                "revised_response": f"safe answer {idx}",
                "source": f"safe-source-{idx}",
            }
        )
    for idx in range(10):
        raw_rows.append(
            {
                "question": f"unsafe question {idx}",
                "thinking_process": f"unsafe reasoning {idx}",
                "reflection": f"unsafe reflection {idx}",
                "assessment": "unsafe",
                "revised_response": f"unsafe answer {idx}",
                "source": f"unsafe-source-{idx}",
            }
        )

    processed_df, drop_reasons = prepare_self_monitor_dataframe(pd.DataFrame(raw_rows), system_prompt=SYSTEM_PROMPT)

    assert not drop_reasons

    train_df_a, test_df_a, strategy_a = split_processed_dataframe(processed_df, val_ratio=0.1, seed=1)
    train_df_b, test_df_b, strategy_b = split_processed_dataframe(processed_df, val_ratio=0.1, seed=1)

    assert strategy_a == "assessment"
    assert strategy_b == "assessment"
    assert len(train_df_a) == 18
    assert len(test_df_a) == 2
    assert test_df_a["assessment"].value_counts().to_dict() == {"safe": 1, "unsafe": 1}
    assert train_df_a["question"].tolist() == train_df_b["question"].tolist()
    assert test_df_a["question"].tolist() == test_df_b["question"].tolist()