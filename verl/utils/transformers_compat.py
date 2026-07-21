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
"""Compatibility helpers for Hugging Face model families and releases."""

from __future__ import annotations

from types import MethodType


def get_auto_model_for_vision2seq():
    """Return the broadest image-text generation auto class available."""
    try:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText
    except ImportError:
        from transformers import AutoModelForVision2Seq

        return AutoModelForVision2Seq


def get_hf_generation_model_class(hf_config):
    """Resolve a generation auto class without hard-coding model families."""
    from transformers import AutoModel, AutoModelForCausalLM

    image_text_class = get_auto_model_for_vision2seq()
    architectures = list(getattr(hf_config, "architectures", None) or [])
    architecture = architectures[0] if architectures else ""
    auto_map = getattr(hf_config, "auto_map", None) or {}

    for auto_class_name, implementation in auto_map.items():
        implementations = implementation if isinstance(implementation, (list, tuple)) else [implementation]
        if architecture and not any(architecture in str(item) for item in implementations):
            continue
        if auto_class_name in {"AutoModelForImageTextToText", "AutoModelForVision2Seq"}:
            return image_text_class
        if auto_class_name == "AutoModelForCausalLM":
            return AutoModelForCausalLM
        if auto_class_name == "AutoModel":
            return AutoModel

    if type(hf_config) in image_text_class._model_mapping.keys():
        return image_text_class
    if type(hf_config) in AutoModelForCausalLM._model_mapping.keys():
        return AutoModelForCausalLM

    if "ForConditionalGeneration" in architecture or "ForVision2Seq" in architecture:
        return image_text_class
    if "ForCausalLM" in architecture:
        return AutoModelForCausalLM
    return AutoModel


def is_gemma3_config(hf_config) -> bool:
    """Whether a config describes either Gemma3 text or image-text generation."""
    model_type = getattr(hf_config, "model_type", None)
    text_model_type = getattr(getattr(hf_config, "text_config", None), "model_type", None)
    return model_type in {"gemma3", "gemma3_text"} or text_model_type == "gemma3_text"


def is_image_text_config(hf_config) -> bool:
    architectures = list(getattr(hf_config, "architectures", None) or [])
    return bool(
        getattr(hf_config, "vision_config", None) is not None
        or any("ForConditionalGeneration" in architecture for architecture in architectures)
    )


def resolve_attn_implementation(hf_config, requested: str | None = None) -> str:
    """Choose a training attention backend, using Gemma3's correct mask path."""
    if requested is not None:
        if is_gemma3_config(hf_config) and requested != "eager":
            raise ValueError(
                "Gemma3 training requires attn_implementation='eager' with transformers 4.51.1. "
                "FlashAttention 2 does not implement Gemma3's training mask correctly."
            )
        return requested
    return "eager" if is_gemma3_config(hf_config) else "flash_attention_2"


def validate_gemma3_training_options(
    hf_config,
    *,
    attn_implementation: str,
    use_remove_padding: bool,
    use_fused_kernels: bool = False,
    use_liger: bool = False,
    ulysses_sequence_parallel_size: int = 1,
) -> None:
    """Reject local fast paths that do not preserve Gemma3's hybrid attention."""
    if not is_gemma3_config(hf_config):
        return
    if attn_implementation != "eager":
        raise ValueError("Gemma3 training must use attn_implementation='eager'.")
    unsupported = []
    if use_remove_padding:
        unsupported.append("use_remove_padding")
    if use_fused_kernels:
        unsupported.append("use_fused_kernels")
    if use_liger:
        unsupported.append("use_liger")
    if ulysses_sequence_parallel_size > 1:
        unsupported.append("ulysses_sequence_parallel_size > 1")
    if unsupported:
        raise ValueError(
            "Gemma3 is not compatible with this verl-base fast path: "
            + ", ".join(unsupported)
            + ". Disable it so the Hugging Face eager hybrid-attention implementation is used."
        )


def patch_gemma3_conditional_causal_mask(model) -> bool:
    """Keep Gemma3 4B causal when PPO requests logits without passing labels.

    Transformers 4.51.1 otherwise treats a label-free full-sequence forward as
    generation prefill. That is unsuitable for token-level SFT/RL log-probability
    computation because response tokens could attend to later response tokens.
    """
    if not is_gemma3_config(model.config) or not is_image_text_config(model.config):
        return False
    if not hasattr(model, "_update_causal_mask") or getattr(model, "_verl_force_causal_mask", False):
        return False

    original_update_causal_mask = model._update_causal_mask

    def _update_causal_mask(
        self,
        attention_mask,
        token_type_ids,
        past_key_values,
        cache_position,
        input_tensor,
        is_training=False,
    ):
        return original_update_causal_mask(
            attention_mask,
            token_type_ids,
            past_key_values,
            cache_position,
            input_tensor,
            is_training=True,
        )

    model._update_causal_mask = MethodType(_update_causal_mask, model)
    model._verl_force_causal_mask = True
    return True
