import importlib.util
from pathlib import Path

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset


_MODULE_PATH = Path(__file__).resolve().parents[3] / "examples" / "data_preprocess" / "self_monitor_sft.py"
_SPEC = importlib.util.spec_from_file_location("self_monitor_sft_for_dataset_test", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)

build_messages = _MODULE.build_messages
build_target_text = _MODULE.build_target_text


class _FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=True, return_tensors="pt", add_generation_prompt=False):
        rendered = "\n".join(f"<{message['role']}>{message['content']}</{message['role']}>" for message in messages)
        if not tokenize:
            return rendered
        token_ids = torch.tensor([[ord(ch) for ch in rendered]], dtype=torch.long)
        return token_ids

    def decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            values = token_ids.flatten().tolist()
        else:
            values = list(token_ids)
        return "".join(chr(value) for value in values if value != self.pad_token_id)


def test_multiturn_sft_dataset_masks_self_monitor_target(tmp_path):
    tokenizer = _FakeTokenizer()
    target_text = build_target_text(
        thinking_process="Reason through the request carefully.",
        reflection="Double-check whether the answer is safe.",
        assessment="safe",
        revised_response="Final answer for the user.",
    )
    messages = build_messages(question="What should I do?", target_text=target_text)

    parquet_path = tmp_path / "self_monitor.parquet"
    pd.DataFrame({"messages": [messages]}).to_parquet(parquet_path)

    dataset = MultiTurnSFTDataset(
        parquet_files=str(parquet_path),
        tokenizer=tokenizer,
        config={"max_length": 2048, "truncation": "error", "multiturn": {"messages_key": "messages"}},
    )

    item = dataset[0]
    assistant_text = tokenizer.decode(item["input_ids"][item["loss_mask"] == 1])
    non_assistant_text = tokenizer.decode(item["input_ids"][item["loss_mask"] == 0])

    assert "<think>" in assistant_text
    assert "<monitor>Double-check whether the answer is safe.</monitor>" in assistant_text
    assert "<label>safe</label>" in assistant_text
    assert "Final answer for the user." in assistant_text
    assert "What should I do?" not in assistant_text
    assert "What should I do?" in non_assistant_text
