# Gemma3 support

This verl-base branch supports `google/gemma-3-1b-it` and
`google/gemma-3-4b-it` with the pinned remote stack:

- `transformers==4.51.1`
- `vllm==0.8.5`
- PyTorch/FSDP or FSDP2 from the existing remote environment

No heavyweight dependency is required in the local development checkout.

## Support matrix

| Path | Gemma3 1B | Gemma3 4B text | Gemma3 4B image |
| --- | --- | --- | --- |
| SFT, FSDP/FSDP2 | yes | yes | yes, single-turn SFT dataset |
| GRPO/RLVR actor + reference | yes | yes | yes |
| Agentic GRPO | yes | yes | yes |
| vLLM rollout | yes | yes | yes, vLLM V0 |
| PPO/GAE critic | separate supported critic required | separate supported critic required | not supported as a Gemma3 critic |
| Megatron backend | not covered by this backport | not covered | not covered |

Agentic GRPO covers `deceptive-roles`, `search`, `deceptive-search`,
`webshop`, and `cheatshop`. Monitor, verdict-monitor, and judge model loading
uses the same auto-model compatibility path.

## Why the settings differ

Gemma3 1B is `Gemma3ForCausalLM`. Gemma3 4B is
`Gemma3ForConditionalGeneration`, even for text-only prompts. The 4B model
therefore needs the image-text auto-model mapping, processor-aware checkpoints,
and vLLM-compatible weight names.

Transformers 4.51.1 also requires eager attention for correct Gemma3 training.
This branch selects it automatically and rejects these incompatible local fast
paths:

- `use_remove_padding=True`
- `use_fused_kernels=True`
- `use_liger=True`
- `ulysses_sequence_parallel_size > 1`

The 4B Hugging Face wrapper normally chooses its training mask by checking
whether `labels` were passed. PPO requests logits without labels, so this branch
patches that model instance to keep full-sequence scoring causal while retaining
bidirectional attention inside image-token blocks.

## Agentic RL

Run any existing task recipe through the Gemma3 wrapper. The second argument can
be either supported model:

```bash
bash examples/gemma3/run_agentic_grpo.sh deceptive-search google/gemma-3-4b-it
bash examples/gemma3/run_agentic_grpo.sh deceptive-roles google/gemma-3-1b-it
bash examples/gemma3/run_agentic_grpo.sh search google/gemma-3-4b-it
bash examples/gemma3/run_agentic_grpo.sh webshop google/gemma-3-4b-it
bash examples/gemma3/run_agentic_grpo.sh cheatshop google/gemma-3-4b-it
```

Extra Hydra overrides are forwarded after the model argument. The wrapper uses
vLLM V0 because vLLM 0.8.5 V1 does not preserve Gemma3's bidirectional image
attention. V0 is also valid for text-only runs.

Vision and projector parameters are frozen in the conservative agentic recipe.
To fine-tune them in a full-parameter image run, override both flags and leave
FSDP `use_orig_params=True`:

```bash
bash examples/gemma3/run_agentic_grpo.sh webshop google/gemma-3-4b-it \
  actor_rollout_ref.model.freeze_vision_tower=False \
  actor_rollout_ref.model.freeze_multi_modal_projector=False
```

For LoRA, keep adapters on the language backbone:

```text
actor_rollout_ref.model.lora_rank=32
actor_rollout_ref.model.target_modules=all-linear
actor_rollout_ref.model.exclude_modules='.*(vision_tower|multi_modal_projector).*'
actor_rollout_ref.model.freeze_vision_tower=True
actor_rollout_ref.model.freeze_multi_modal_projector=True
```

## SFT

Text SFT uses the existing parquet schema. For image SFT, add an `images` column
containing one or more image records and place matching `<image>` markers in the
prompt. If no marker is present, image segments are prepended to the user turn.
Multimodal samples are not truncated because cutting image tokens invalidates
the image-feature alignment; increase `data.max_length` instead.

```bash
python -m verl.trainer.fsdp_sft_trainer \
  model.partial_pretrain=google/gemma-3-4b-it \
  model.attn_implementation=eager \
  model.freeze_vision_tower=True \
  model.freeze_multi_modal_projector=True \
  model.fsdp_config.use_orig_params=True \
  use_remove_padding=False \
  ulysses_sequence_parallel_size=1 \
  data.train_files=/path/to/train.parquet \
  data.val_files=/path/to/validation.parquet
```

## Remote validation

Run these checks on the remote GPU server after syncing the checkout. They load
real weights; they are intentionally not part of local CI.

```bash
python tests/e2e/gemma3/check_model_stack.py google/gemma-3-1b-it
python tests/e2e/gemma3/check_model_stack.py google/gemma-3-4b-it
```

Then launch a short task run by appending task-specific reductions such as
`trainer.total_training_steps=1`, small batch sizes, and `trainer.test_freq=-1`
to `run_agentic_grpo.sh`.

Model access must already be accepted and cached when the task scripts enable
Hugging Face offline mode.

## References

- [Gemma3 model documentation](https://huggingface.co/docs/transformers/v4.51.1/en/model_doc/gemma3)
- [vLLM 0.8.5 supported models](https://docs.vllm.ai/en/v0.8.5/models/supported_models.html)
- [Upstream veRL Transformers compatibility helper](https://github.com/verl-project/verl/blob/main/verl/utils/transformers_compat.py)
