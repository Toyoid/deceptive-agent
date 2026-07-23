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
import gc
import os
from importlib.metadata import version

import torch
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from verl.utils.fsdp_utils import get_fsdp_wrap_policy
from verl.utils.torch_functional import entropy_from_logits, entropy_from_logits_with_chunking
from verl.utils.transformers_compat import (
    get_hf_generation_model_class,
    patch_gemma3_conditional_causal_mask,
    resolve_attn_implementation,
)


def validate_generation(tokenizer, token_ids, backend):
    raw_output = tokenizer.decode(token_ids, skip_special_tokens=False)
    output = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    if not output or "ok" not in output.lower() or "<unused" in raw_output:
        raise RuntimeError(f"{backend} generated an invalid smoke-test response: {raw_output!r}")
    return output


def validate_padded_backward(model, tokenizer):
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "Reply with OK."}],
            add_generation_prompt=True,
            tokenize=False,
        ),
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "Read this longer request, then reply with exactly OK."}],
            add_generation_prompt=True,
            tokenize=False,
        ),
    ]
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        batch = tokenizer(prompts, padding=True, return_tensors="pt")
    finally:
        tokenizer.padding_side = original_padding_side
    batch = {key: value.cuda() for key, value in batch.items()}

    model.train()
    outputs = model(**batch, use_cache=False)
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = batch["input_ids"][:, 1:]
    valid_tokens = batch["attention_mask"][:, :-1].bool() & batch["attention_mask"][:, 1:].bool()
    loss = torch.nn.functional.cross_entropy(
        shift_logits[valid_tokens].float(),
        shift_labels[valid_tokens],
    )
    loss.backward()

    bad_gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    if not torch.isfinite(loss) or bad_gradients:
        raise RuntimeError(
            f"Padded backward produced non-finite values: loss={loss.item()}, "
            f"parameters={bad_gradients[:10]}"
        )
    print(f"PASS padded-backward model={model.__class__.__name__} loss={loss.item():.6f}")
    model.zero_grad(set_to_none=True)
    model.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Gemma3 Hugging Face ID or local checkpoint path")
    parser.add_argument("--check-vllm", action="store_true", help="Compare a greedy text generation with vLLM V0")
    parser.add_argument("--check-backward", action="store_true", help="Check a left-padded training forward and backward")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
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
    wrap_policy = get_fsdp_wrap_policy(
        model,
        {
            "min_num_params": 0,
            "transformer_layer_cls_to_wrap": "Gemma3DecoderLayer",
        },
    )
    assert wrap_policy is not None

    test_logits = torch.randn(2, 3, 257, device="cuda", dtype=torch.bfloat16)
    expected_entropy = entropy_from_logits(test_logits.float())
    chunked_entropy = entropy_from_logits_with_chunking(test_logits, chunk_size=2)
    torch.testing.assert_close(chunked_entropy, expected_entropy, rtol=1e-4, atol=1e-4)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.check_backward:
        validate_padded_backward(model, tokenizer)
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

    text_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Reply with exactly OK."}],
        add_generation_prompt=True,
        return_tensors="pt",
    ).cuda()
    with torch.no_grad():
        hf_output_ids = model.generate(
            input_ids=text_prompt,
            attention_mask=torch.ones_like(text_prompt),
            do_sample=False,
            max_new_tokens=16,
        )[0, text_prompt.shape[-1] :]
    hf_output = validate_generation(tokenizer, hf_output_ids, "Hugging Face")

    print(
        f"PASS hf model={args.model} class={model.__class__.__name__} "
        f"logits={tuple(logits.shape)} output={hf_output!r}"
    )

    if args.check_vllm:
        del logits, model
        gc.collect()
        torch.cuda.empty_cache()

        os.environ["VLLM_USE_V1"] = "0"
        from vllm import LLM, SamplingParams

        engine = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype="bfloat16",
            enforce_eager=True,
            enable_chunked_prefill=False,
            enable_prefix_caching=False,
            max_model_len=512,
        )
        prompt_ids = text_prompt[0].cpu().tolist()
        outputs = engine.generate(
            [{"prompt_token_ids": prompt_ids}],
            SamplingParams(temperature=0, max_tokens=16),
            use_tqdm=False,
        )
        vllm_output = validate_generation(tokenizer, outputs[0].outputs[0].token_ids, "vLLM")
        print(f"PASS vllm-v0 model={args.model} output={vllm_output!r}")


if __name__ == "__main__":
    main()
