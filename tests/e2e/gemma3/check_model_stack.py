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

"""Remote-GPU smoke check for the pinned Gemma3 Transformers stack."""

import argparse
from importlib.metadata import version

import torch
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from verl.utils.fsdp_utils import get_fsdp_wrap_policy
from verl.utils.transformers_compat import (
    get_hf_generation_model_class,
    patch_gemma3_conditional_causal_mask,
    resolve_attn_implementation,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Gemma3 Hugging Face ID or local checkpoint path")
    args = parser.parse_args()

    expected_versions = {"transformers": "4.51.1", "vllm": "0.8.5"}
    for package, expected in expected_versions.items():
        installed = version(package)
        if installed != expected:
            raise RuntimeError(f"Expected {package}=={expected}, found {installed}.")

    config = AutoConfig.from_pretrained(args.model)
    if config.model_type not in {"gemma3", "gemma3_text"}:
        raise ValueError(f"Expected a Gemma3 checkpoint, found model_type={config.model_type!r}.")
    model_class = get_hf_generation_model_class(config)
    model = model_class.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation=resolve_attn_implementation(config),
    ).cuda()
    patch_gemma3_conditional_causal_mask(model)
    wrap_policy = get_fsdp_wrap_policy(model, {"min_num_params": 0})
    assert wrap_policy is not None

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if config.model_type == "gemma3":
        from PIL import Image

        processor = AutoProcessor.from_pretrained(args.model)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Reply with the dominant color."},
                ],
            }
        ]
        prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(
            text=[prompt_text],
            images=[Image.new("RGB", (64, 64), color="red")],
            return_tensors="pt",
        )
        inputs = {key: value.cuda() for key, value in inputs.items()}
    else:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Reply with OK."}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).cuda()
        inputs = {"input_ids": prompt, "attention_mask": torch.ones_like(prompt)}

    with torch.no_grad():
        logits = model(**inputs, use_cache=False).logits
    assert logits.shape[:2] == inputs["input_ids"].shape
    print(f"PASS model={args.model} class={model.__class__.__name__} logits={tuple(logits.shape)}")


if __name__ == "__main__":
    main()
