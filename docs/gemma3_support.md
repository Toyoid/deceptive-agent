# Gemma3 support

This verl-base branch supports `google/gemma-3-1b-it` and
`google/gemma-3-4b-it` with the pinned remote stack:

- `transformers==4.51.1`
- `vllm==0.8.5`
- PyTorch/FSDP or FSDP2 from the existing remote environment

No heavyweight dependency is required in the local development checkout.

The vLLM pin predates upstream's complete Gemma3 sliding-attention cleanup.
This branch therefore uses a conservative rollout profile: vLLM V0, eager
execution, no chunked prefill, and no prefix caching. Do not remove those
settings without first comparing greedy Hugging Face and vLLM outputs on the
remote machine. vLLM 0.10.1 contains the later upstream sliding-attention fix,
but upgrading to it also requires a coordinated Torch 2.7.1 and Transformers
4.55+ migration; it is not a safe one-package upgrade for this fork.

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
vLLM V0 because this branch's external FSDP weight synchronization and
sleep/wake lifecycle are validated on V0. Prefix caching is disabled because
cache state can become stale across that lifecycle and produce corrupted first
batches. `main_ppo` applies V0 automatically for recognized Gemma3 model paths,
including direct Python launches. It also preloads the vLLM model from
`safetensors` instead of random `dummy_dtensor` weights and verifies that every
vLLM parameter is populated by each full-parameter FSDP synchronization.

Gemma3's 262k-token vocabulary also makes the ordinary categorical-entropy
formula unusually expensive. This branch automatically computes entropy in
256-row chunks for vocabularies with at least 131k tokens. Entropy
regularization defaults to zero, matching upstream verl; when explicitly
enabled, chunked entropy is activation-checkpointed automatically during the
policy update.

Transformers 4.51.1 builds the 4B conditional model's outer causal mask from
the wrapper dtype. Under FSDP mixed precision that dtype can differ from the
BF16 text input dtype, turning the FP32 mask sentinel into `-inf`. The local
compatibility patch converts the mask to the text compute dtype while keeping
the sentinel finite. This prevents left-padded prompt rows from poisoning the
backward pass with non-finite gradients.

Vision and projector parameters are frozen in the conservative agentic recipe.
The `exclude_modules` setting is a LoRA target filter and has no effect in a
full-parameter run; the two freeze flags are sufficient here.
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

The self-monitor text SFT recipe preprocesses `PKU-Alignment/self-monitor` with
Gemma3's own chat template and trains only assistant-turn targets:

```bash
bash examples/sft/self_monitor/run_gemma3_4b.sh 8
```

It defaults to 2048 tokens because Gemma3 training uses eager attention and a
262k-token output vocabulary in this pinned stack. Increase the length only
after a one-step memory check, for example with `MAX_LENGTH=4096`.

```bash
python -m verl.trainer.fsdp_sft_trainer \
  model.partial_pretrain=google/gemma-3-4b-it \
  model.attn_implementation=eager \
  model.freeze_vision_tower=True \
  model.freeze_multi_modal_projector=True \
  model.strategy=fsdp \
  model.fsdp_config.use_orig_params=True \
  +model.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=Gemma3DecoderLayer \
  use_remove_padding=False \
  ulysses_sequence_parallel_size=1 \
  data.train_files=/path/to/train.parquet \
  data.val_files=/path/to/validation.parquet
```

## Remote validation

Run these checks on the remote GPU server after syncing the checkout. They load
real weights; they are intentionally not part of local CI.

```bash
python tests/e2e/gemma3/check_model_stack.py google/gemma-3-1b-it --check-backward --check-vllm
python tests/e2e/gemma3/check_model_stack.py google/gemma-3-4b-it --check-backward --check-vllm
```

Then launch a short task run by appending task-specific reductions such as
`trainer.total_training_steps=1`, small batch sizes, and `trainer.test_freq=-1`
to `run_agentic_grpo.sh`.

Model access must already be accepted and cached when the task scripts enable
Hugging Face offline mode.

## References

- [Gemma3 model documentation](https://huggingface.co/docs/transformers/v4.51.1/en/model_doc/gemma3)
- [vLLM 0.8.5 supported models](https://docs.vllm.ai/en/v0.8.5/models/supported_models.html)
- [vLLM Gemma3 accuracy report](https://github.com/vllm-project/vllm/issues/17689)
- [vLLM sliding-attention fix](https://github.com/vllm-project/vllm/pull/21927)
- [veRL 0.4.1 memory-optimization notes](https://github.com/verl-project/verl/discussions/2225)
- [Upstream veRL Transformers compatibility helper](https://github.com/verl-project/verl/blob/main/verl/utils/transformers_compat.py)
