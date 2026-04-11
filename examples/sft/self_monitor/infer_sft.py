#!/usr/bin/env python3
"""Run a quick manual prompt against a saved self-monitor SFT checkpoint."""

import argparse

import torch
from transformers import AutoModelForCausalLM

from verl.utils.tokenizer import hf_tokenizer


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful and harmless assistant.\n"
    "Before answering, think step by step, and your response must follow this format:\n"
    "<think>\n"
    "your reasoning here\n"
    "</think>\n"
    "your final answer"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a saved SFT checkpoint and run a single prompt.")
    parser.add_argument("--checkpoint", required=True, help="Path to the saved Hugging Face checkpoint directory.")
    parser.add_argument("--prompt", default=None, help="User prompt to send to the model. If omitted, you will be prompted in the terminal.")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="System prompt to prepend before the user prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature. Use 0 for greedy decoding.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p nucleus sampling value when temperature > 0.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True when loading model/tokenizer.")
    return parser.parse_args()


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_and_tokenizer(checkpoint: str, trust_remote_code: bool):
    tokenizer = hf_tokenizer(checkpoint, trust_remote_code=trust_remote_code)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    return model, tokenizer


def build_inputs(tokenizer, system_prompt: str, user_prompt: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if isinstance(model_inputs, torch.Tensor):
        input_ids = model_inputs
    else:
        input_ids = torch.tensor(model_inputs, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return input_ids.to(device), attention_mask.to(device)


def generate_response(model, tokenizer, input_ids: torch.Tensor, attention_mask: torch.Tensor, max_new_tokens: int, temperature: float, top_p: float) -> str:
    do_sample = temperature > 0
    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.no_grad():
        generated = model.generate(**generation_kwargs)

    generated_tokens = generated[0, input_ids.shape[-1] :]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    prompt = args.prompt if args.prompt is not None else input("Enter your prompt: ").strip()
    if not prompt:
        raise ValueError("Prompt must not be empty.")

    device = pick_device()
    model, tokenizer = load_model_and_tokenizer(args.checkpoint, trust_remote_code=args.trust_remote_code)
    model.to(device)
    model.eval()

    input_ids, attention_mask = build_inputs(
        tokenizer=tokenizer,
        system_prompt=args.system_prompt,
        user_prompt=prompt,
        device=device,
    )
    response = generate_response(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(response)


if __name__ == "__main__":
    main()