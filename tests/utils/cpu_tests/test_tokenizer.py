import importlib.util
from pathlib import Path


_TOKENIZER_MODULE_PATH = Path(__file__).resolve().parents[3] / "verl" / "utils" / "tokenizer.py"
_SPEC = importlib.util.spec_from_file_location("verl_utils_tokenizer_for_test", _TOKENIZER_MODULE_PATH)
_TOKENIZER_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(_TOKENIZER_MODULE)

_patch_apply_chat_template_defaults = _TOKENIZER_MODULE._patch_apply_chat_template_defaults


class _SupportsThinkingTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, enable_thinking=True):
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "enable_thinking": enable_thinking,
            }
        )
        return enable_thinking


class _NoThinkingTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False):
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
            }
        )
        return "ok"


def test_patch_disables_thinking_by_default():
    tokenizer = _SupportsThinkingTokenizer()
    _patch_apply_chat_template_defaults(tokenizer)

    result = tokenizer.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False)

    assert result is False
    assert tokenizer.calls[0]["enable_thinking"] is False


def test_patch_preserves_explicit_enable_thinking_override():
    tokenizer = _SupportsThinkingTokenizer()
    _patch_apply_chat_template_defaults(tokenizer)

    result = tokenizer.apply_chat_template([{"role": "user", "content": "hi"}], enable_thinking=True)

    assert result is True
    assert tokenizer.calls[0]["enable_thinking"] is True


def test_patch_falls_back_cleanly_for_tokenizers_without_enable_thinking():
    tokenizer = _NoThinkingTokenizer()
    _patch_apply_chat_template_defaults(tokenizer)

    result = tokenizer.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False)

    assert result == "ok"
    assert tokenizer.calls == [{"messages": [{"role": "user", "content": "hi"}], "tokenize": False}]


def test_patch_supports_per_instance_chat_template_defaults():
    tokenizer = _SupportsThinkingTokenizer()
    _patch_apply_chat_template_defaults(tokenizer, default_kwargs={"enable_thinking": True})

    result = tokenizer.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False)

    assert result is True
    assert tokenizer.calls[0]["enable_thinking"] is True
