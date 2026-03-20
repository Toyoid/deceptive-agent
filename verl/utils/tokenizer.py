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
"""Utils for tokenization."""

import functools
import inspect
import warnings

__all__ = ["hf_tokenizer", "hf_processor"]

_DEFAULT_APPLY_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}


def _normalize_apply_chat_template_default_kwargs(default_kwargs):
    if default_kwargs is None:
        return dict(_DEFAULT_APPLY_CHAT_TEMPLATE_KWARGS)
    return dict(default_kwargs)


def _filter_supported_apply_chat_template_kwargs(apply_chat_template, kwargs):
    if not kwargs:
        return {}

    try:
        signature = inspect.signature(apply_chat_template)
    except (TypeError, ValueError):
        return dict(kwargs)

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return dict(kwargs)

    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _patch_apply_chat_template_defaults(tokenizer_or_processor, default_kwargs=None):
    """Apply default chat-template kwargs to a tokenizer/processor instance.

    Some models such as Qwen3 accept ``enable_thinking`` in ``apply_chat_template`` and
    enable thinking mode by default. We want the repo behavior to stay stable unless a
    caller explicitly opts in, so we inject model-specific default kwargs only when the
    method supports them. Explicit callsite kwargs always win.
    """
    tokenizer_or_processor._verl_apply_chat_template_default_kwargs = (
        _normalize_apply_chat_template_default_kwargs(default_kwargs)
    )

    apply_chat_template = getattr(tokenizer_or_processor, "apply_chat_template", None)
    if apply_chat_template is None:
        return tokenizer_or_processor

    # Avoid stacking wrappers if the same object is patched more than once. We still keep
    # the latest per-instance defaults via the attribute set above.
    if getattr(apply_chat_template, "_verl_disable_thinking_patched", False):
        return tokenizer_or_processor

    @functools.wraps(apply_chat_template)
    def wrapped_apply_chat_template(*args, **kwargs):
        default_chat_template_kwargs = getattr(
            tokenizer_or_processor,
            "_verl_apply_chat_template_default_kwargs",
            {},
        )
        injected_kwargs = {
            key: value
            for key, value in default_chat_template_kwargs.items()
            if key not in kwargs
        }
        supported_injected_kwargs = _filter_supported_apply_chat_template_kwargs(
            apply_chat_template,
            injected_kwargs,
        )
        if not supported_injected_kwargs:
            return apply_chat_template(*args, **kwargs)

        try:
            return apply_chat_template(*args, **supported_injected_kwargs, **kwargs)
        except TypeError as exc:
            if not any(key in str(exc) for key in supported_injected_kwargs):
                raise
            return apply_chat_template(*args, **kwargs)

    wrapped_apply_chat_template._verl_disable_thinking_patched = True
    tokenizer_or_processor.apply_chat_template = wrapped_apply_chat_template
    return tokenizer_or_processor


def set_pad_token_id(tokenizer):
    """Set pad_token_id to eos_token_id if it is None.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be set.

    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}", stacklevel=1)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f"tokenizer.pad_token is None. Now set to {tokenizer.eos_token}", stacklevel=1)


def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, apply_chat_template_default_kwargs=None, **kwargs):
    """Create a huggingface pretrained tokenizer which correctness handles eos and pad tokens.

    Args:

        name (str): The name of the tokenizer.
        correct_pad_token (bool): Whether to correct the pad token id.
        correct_gemma2 (bool): Whether to correct the gemma2 tokenizer.

    Returns:

        transformers.PreTrainedTokenizer: The pretrained tokenizer.

    """
    from transformers import AutoTokenizer

    if correct_gemma2 and isinstance(name_or_path, str) and "gemma-2-2b-it" in name_or_path:
        # the EOS token in gemma2 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        warnings.warn("Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.", stacklevel=1)
        kwargs["eos_token"] = "<end_of_turn>"
        kwargs["eos_token_id"] = 107
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    if correct_pad_token:
        set_pad_token_id(tokenizer)
    _patch_apply_chat_template_defaults(
        tokenizer,
        default_kwargs=apply_chat_template_default_kwargs,
    )
    return tokenizer


def hf_processor(name_or_path, apply_chat_template_default_kwargs=None, **kwargs):
    """Create a huggingface processor to process multimodal data.

    Args:
        name_or_path (str): The name of the processor.

    Returns:
        transformers.ProcessorMixin: The pretrained processor.
    """
    from transformers import AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except Exception:
        processor = None
    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/auto/processing_auto.py#L344
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    if processor is not None:
        _patch_apply_chat_template_defaults(
            processor,
            default_kwargs=apply_chat_template_default_kwargs,
        )
    return processor
