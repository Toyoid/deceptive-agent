# MonitorInfer Gibberish Output Investigation

This note documents the MonitorInfer corruption issue, summarizes the observed symptoms, explains KV cache and prefix caching in vLLM, and outlines the current hypothesis plus mitigations.

## Symptoms

Observed with MonitorInfer outputs such as:

```
<critique>
></crit
user <<crit
icine> <[
critique>DE deception] The The assistant fabricated the user provided. The the assistant intentionally misides the user by the provid
ed is not a standard practice and it and and and and and and and and the user.
```

The issue manifests under specific combinations of configurations:

| Config | Failing Values | Working Values |
|--------|---------------|----------------|
| `val_batch_size` | 64, 40, 35, 30, 29 | 6, 15, 25, 27 |
| `n_gpus_per_node_monitor` | 1, 2 | 3 |
| `enable_prefix_caching` | True | False |
| Monitor model | Qwen2.5-7B-Instruct | Qwen2.5-3B-Instruct |

### Not affected
- `enable_chunked_prefill` (True/False)
- `load_format` ('safetensors' vs 'dummy_dtensor')
- `fsdp_config.model_dtype` ('bfloat16' vs 'float32')

### Warm-up effect
- With actor + monitor both `enable_prefix_caching=True`, 2 GPUs each, 7B trainable + ref models:
  - `val_batch_size=56` (two batches: 56 + 4)
  - Actor: batch 1 and batch 2 normal
  - Monitor: batch 1 corrupted, batch 2 normal

### Prompt structure experiments
All runs below: `enable_prefix_caching=True`, `n_gpus_per_node_monitor=2`, `n_gpus_per_node=2`, `val_batch_size=64`, no prompts filtered.

- Shorter monitor prompt -> actor and monitor both normal.
- Keep simple monitor prompt, add long ANTI_MONITOR prompt into actor system prompt -> actor and monitor both normal.
- Long monitor prompt at the beginning -> monitor corrupted.
- Long monitor prompt moved *after* user content -> monitor normal.

## Background: KV cache and prefix caching in vLLM

### KV cache in autoregressive decoding
In transformer-based LLMs, generating each token requires computing **attention** over all previous tokens. The attention mechanism uses three vectors for each token:
- **K (Key)**: Represents the token's "searchable" features
- **V (Value)**: Represents the token's "content" features  
- **Q (Query)**: What the current token is "looking for"

```
Token Generation Flow:
┌─────────────────────────────────────────────────────────────────┐
│ Prompt: "What is the capital of France?"                        │
│                                                                 │
│ Step 1: Process all prompt tokens                               │
│   "What" → K₁, V₁                                               │
│   "is"   → K₂, V₂                                               │
│   "the"  → K₃, V₃                                               │
│   ...    → ...                                                  │
│                                                                 │
│ Step 2: Generate "Paris"                                        │
│   Q_new attends to [K₁,K₂,K₃,...] and retrieves [V₁,V₂,V₃,...]  │
│                                                                 │
│ Step 3: Generate "is"                                           │
│   Q_new attends to [K₁,K₂,K₃,...,K_Paris] (reuse previous K,V!) │
└─────────────────────────────────────────────────────────────────┘
```

**KV Cache** stores the K and V vectors for all processed tokens in GPU memory. Without it, generating each new token would require recomputing K,V for ALL previous tokens - extremely wasteful!

### vLLM prefix caching
Prefix caching extends KV caching across requests: if many requests share a *common prefix*, vLLM stores the KV blocks for that shared prefix in a prefix cache. Subsequent requests can reuse those cached blocks, reducing prefill compute.

Conceptually:

```
Shared prefix (system + template) -> cached once
Request A suffix -> compute only suffix
Request B suffix -> compute only suffix
```

```
Example: Monitor model processing multiple samples
┌─────────────────────────────────────────────────────────────────┐
│ Sample 1: "[System prompt] [User query A] [Response A]"         │
│ Sample 2: "[System prompt] [User query B] [Response B]"         │
│ Sample 3: "[System prompt] [User query C] [Response C]"         │
│                                                                 │
│ Without prefix caching:                                         │
│   - Compute K,V for [System prompt] 3 times ❌                 
│                                                                 │
│ With prefix caching:                                            │
│   - Compute K,V for [System prompt] once                        │
│   - Reuse cached K,V for samples 2 and 3 ✅                     
│   - Massive speedup for batched inference!                      │
└─────────────────────────────────────────────────────────────────┘
```

Implementation details (high level):
- vLLM hashes the prefix and builds a trie of cached blocks.
- KV cache blocks are allocated in fixed-size chunks, and prefix cache entries point to these blocks.
- Prefix cache reuse is only possible when the longest common prefix (LCP) across requests is long enough to span multiple blocks.

### Why prompt position matters
Prefix caching reuses *prefixes* only. If long template text is placed at the **start** of the prompt, the shared prefix is long and prefix caching is heavily exercised. If the same text is moved **after** user-specific content, the shared prefix becomes short and prefix caching yields little reuse.

This aligns with the experiments:
- Long monitor prompt at the beginning -> large shared prefix -> corruption.
- Same prompt moved after user content -> small shared prefix -> normal output.

## Hypothesis: prefix cache + sleep/wake + FSDP weight sync bug

### Where the path occurs
Monitor and actor use vLLM rollout with a FSDP sharding manager:
- `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`
- `verl/workers/sharding_manager/fsdp_vllm.py`

The relevant sequence is:
1. vLLM engine created with `enable_sleep_mode=True` and `enable_prefix_caching=True`.
2. Engine immediately sleeps (`sleep(level=1)`), freeing KV cache memory.
3. On each rollout:
   - `wake_up(tags=["weights"])`
   - FSDP weight sync (`update_params`)
   - `wake_up(tags=["kv_cache"])`
   - `reset_prefix_cache()`

Notes already in code mention that `reset_prefix_cache()` does not reliably fix corruption.

### Hypothesized failure mode
- The vLLM V1 prefix cache maintains multiple internal structures (trie, block allocator, scheduler tracking).
- In the sleep/wake + external weight sync path, these structures are not fully reset, leaving inconsistent cache state.
- When prefix caching is *heavily exercised* (large shared prefixes), the corrupted cache produces gibberish outputs.

This matches:
- `enable_prefix_caching=False` -> normal output.
- Batch-size and GPU-count dependence -> changes KV block layout and allocator behavior.
- Monitor more affected -> its prompts share a long, identical prefix (system + template), so prefix caching is frequently reused.
- Actor mostly unaffected -> prompt diverges earlier due to per-sample user content, so prefix caching is less active.

### Lifecycle Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         generate_sequences() called                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ with self.rollout_sharding_manager:  ───► __enter__() called                │
│   ├── wake_up(tags=["weights"])      # Load model weights to GPU            │
│   ├── update_params()                # Sync FSDP weights to vLLM            │
│   ├── wake_up(tags=["kv_cache"])     # Allocate KV cache                    │
│   └── reset_prefix_cache()           # ⚠️ DOESN'T FULLY CLEAR STATE         
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ rollout.generate_sequences(prompts)                                         │
│   └── ⚠️ Prefix cache initialization bug may cause corrupted outputs             
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ __exit__() called                                                           │
│   ├── reset_prefix_cache()           # ⚠️ DOESN'T FULLY CLEAR STATE         
│   └── sleep(level=1)                 # Offload weights, free KV cache       │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Why batch size, GPU count, and model size matter

These knobs affect KV cache block layout and prefix cache allocation patterns:
- Larger batch sizes allocate more KV blocks, increasing probability of hitting the buggy allocator path.
- Different `n_gpus_per_node_monitor` changes world size and padding, altering the effective batch size and cache layout.
- Larger models have larger KV tensors, affecting block usage and memory fragmentation.

The non-linear behavior (e.g., 29 fails but 27 works) is consistent with block allocation thresholds.

## Mitigations / Workarounds

### 1) Disable prefix caching for monitor (most reliable)

```
monitor_rollout_ref.rollout.enable_prefix_caching=False
```

This uses the standard KV cache path without the prefix cache, avoiding the bug entirely.

### 2) Reduce shared prefix length

If prefix caching must remain enabled, reduce the length of shared prefixes in monitor prompts:
- Use a shorter monitor system prompt.
- Move long template text after user-specific content (so it is not part of the shared prefix).

### 3) Adjust batch size or GPU count

Changing `val_batch_size` or `n_gpus_per_node_monitor` can move the allocation pattern away from the buggy layout, but this is not stable or guaranteed.

### 4) Avoid sleep/wake for monitor

A more invasive code change is to avoid `sleep()` for monitor vLLM (or avoid `enable_sleep_mode=True`) so the prefix cache is never reinitialized in the problematic sequence. This requires patching the vLLM rollout/sharding manager code path.

## Suggested next experiments

- Log the longest common prefix (LCP) length across a monitor batch to confirm that corruption correlates with LCP length.
- Record padded batch sizes (after `pad_dataproto_to_divisor`) for the monitor rollout.
- Compare LCP length distribution between actor and monitor prompts.

## References in repo

- `agent_system/multi_turn_rollout/rollout_loop.py`
- `agent_system/environments/prompts/monitor_prompt.py`
- `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`
- `verl/workers/sharding_manager/fsdp_vllm.py`

## Appendix: Prompt asking AI to analyze the root cause

I have an issue, There is a strange issue:
MonitorInfer will output gibberish sentences such as:
```
<critique>                                                                                                                           
></crit                                                                                                                              
user <<crit                                                                                                                               
icine>                                                              <[       
critique>DE deception] The The assistant fabricated the user provided. The the assistant intentionally misides the user by the provid
ed is not a standard practice and it and and and and and and and and the user. If by the evidence. is a not a standard the user s and the user.
```

I did a lot of debugging, I found that in several settings, the MonitorInfer model will output gibberish, related to configs including `val_batch_size`, `n_gpus_per_node_monitor`, `enable_prefix_caching`, and monitor model.

### About `val_batch_size`
- Setting `n_gpus_per_node_monitor`=1, and `enable_prefix_caching`=True
- When setting `val_batch_size`=64, 40 35, 30, or 29, MonitorInfer will output gibberish as shown above. 
- However, if I set val_batch_size as 6, 15, 25, 27 for MonitorInfer, the output will be normal as:
```
<critique>HALLUCINATION The assistant fabricates a scenario where it was distracted by a phone, which was not mentioned in the original history.</critique>
```

### About `n_gpus_per_node_monitor`
- With `val_batch_size`=64, and `enable_prefix_caching`=True
- When setting `n_gpus_per_node_monitor`=1 or 2, MonitorInfer will output corrupted answers. 
- However, for `n_gpus_per_node_monitor`=3, the output will be normal.

### About `enable_prefix_caching`
- With `val_batch_size`=64, `n_gpus_per_node_monitor`=1
- When setting `enable_prefix_caching`=True, MonitorInfer will output corrupted answers.
- However, for `enable_prefix_caching`=False, the output will be normal.
One AI suggested that `enable_prefix_caching`=True might be the absolute cause of corrupted answers, the AI tried some modifications and left some notes in the code comments. It said that the root cause is a vLLM V1 engine bug where reset_prefix_cache() doesn't fully clear prefix cache state when combined with sleep mode and FSDP weight sync. But I am not sure and suspicious about whether this is the correct analyze.

### About monitor model
- With `val_batch_size`=64, `n_gpus_per_node_monitor`=2, `enable_prefix_caching`=True
- When setting monitor model as `Qwen/Qwen2.5-7B-Instruct`, MonitorInfer will output corrupted answers.
for example:
```
<critique>
H

user: No
user
No issues identified.
```

- However, when setting monitor model as `Qwen/Qwen2.5-3B-Instruct`, the output will be fluent (no unmeaningful signals).

### other configs tried but not relevant
- Changing `enable_chunked_prefill` between True and False does not affect the gibberish output.
- Changing `load_format` between 'safetensors' and 'dummy_dtensor' does not make any difference.
- changing `fsdp_config.model_dtype` between 'bfloat16' and 'float32' does not make any difference.


## New Observations

### Experiment 1: Both Actor and Monitor with `enable_prefix_caching=True`, 2 GPUs each, tranable model with 7B parameters, and ref models.
- **val_dataset_size**: 60
- **val_batch_size**: 56
- **Result**: Validation has 2 batches (frist batch size 56, second batch size 4)
  - **Actor**: Both batch 1 and batch 2 outputs are **NORMAL** ✅
  - **Monitor**: Batch 1 output is **CORRUPTED** ❌, Batch 2 output is **NORMAL** ✅

### Experiment 2: Actor with `enable_prefix_caching=True`, Monitor with `enable_prefix_caching=False`
- Same configuration as above
- **Result**: 
  - **Actor**: All outputs **NORMAL** ✅
  - **Monitor**: All outputs **NORMAL** ✅

### Key Insights from New Experiments

1. **Actor with prefix caching works perfectly** - no corruption in any batch
2. **Monitor with prefix caching corrupts only the FIRST batch (batch_size 56)** - second batch is normal (batch_size 4)
3. **Actor and Monitor use separate GPU pools** - so direct GPU memory sharing is not the issue
4. **The "warm-up" effect**: After processing batch 1 (even with corruption), batch 2 works correctly

### Task
Based on the facts above, take a deep dive into the codebase and analyze what is the root cause of MonitorInfer outputting gibberish sentences? 
important entry files may include: examples/ppo_trainer/run_deceptive_roles.sh, verl/trainer/config/ppo_trainer.yaml , verl/trainer/main_ppo.py , verl/trainer/ppo/ray_trainer.py , agent_system/multi_turn_rollout/rollout_loop.py , verl/workers/fsdp_workers.py , verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py
 