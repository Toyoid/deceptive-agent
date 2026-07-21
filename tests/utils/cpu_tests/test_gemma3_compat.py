# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[3] / "verl" / "utils" / "transformers_compat.py"
SPEC = importlib.util.spec_from_file_location("transformers_compat_under_test", MODULE_PATH)
compat = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compat)


class FakeConfig:
    def __init__(self, model_type, architecture, vision=False):
        self.model_type = model_type
        self.architectures = [architecture]
        self.vision_config = object() if vision else None


def test_gemma3_defaults_to_eager_attention():
    config = FakeConfig("gemma3_text", "Gemma3ForCausalLM")
    assert compat.is_gemma3_config(config)
    assert compat.resolve_attn_implementation(config) == "eager"


def test_gemma3_rejects_flash_attention_training():
    config = FakeConfig("gemma3", "Gemma3ForConditionalGeneration", vision=True)
    with pytest.raises(ValueError, match="requires attn_implementation='eager'"):
        compat.resolve_attn_implementation(config, "flash_attention_2")


@pytest.mark.parametrize(
    "options, expected",
    [
        ({"use_remove_padding": True}, "use_remove_padding"),
        ({"use_fused_kernels": True}, "use_fused_kernels"),
        ({"use_liger": True}, "use_liger"),
        ({"ulysses_sequence_parallel_size": 2}, "ulysses_sequence_parallel_size"),
    ],
)
def test_gemma3_rejects_unsupported_fast_paths(options, expected):
    config = FakeConfig("gemma3", "Gemma3ForConditionalGeneration", vision=True)
    defaults = {
        "attn_implementation": "eager",
        "use_remove_padding": False,
        "use_fused_kernels": False,
        "use_liger": False,
        "ulysses_sequence_parallel_size": 1,
    }
    defaults.update(options)
    with pytest.raises(ValueError, match=expected):
        compat.validate_gemma3_training_options(config, **defaults)


def test_non_gemma_model_keeps_existing_default():
    config = FakeConfig("qwen2", "Qwen2ForCausalLM")
    assert compat.resolve_attn_implementation(config) == "flash_attention_2"


def test_conditional_model_patch_forces_causal_scoring_without_labels():
    class FakeModel:
        config = FakeConfig("gemma3", "Gemma3ForConditionalGeneration", vision=True)

        def _update_causal_mask(self, *args, is_training=False):
            return is_training

    model = FakeModel()
    assert compat.patch_gemma3_conditional_causal_mask(model)
    assert model._update_causal_mask(None, None, None, None, None, is_training=False)
    assert not compat.patch_gemma3_conditional_causal_mask(model)
