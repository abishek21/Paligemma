# Understanding LLM Inference: Prefill, Decode & KV Cache

This guide explains the two stages of autoregressive generation, how they map to
**this** PaliGemma codebase, which metrics matter, and how to measure them with
`benchmark_inference.py`. Stage 2 extends the study to production serving with
**vLLM** and concurrent users.

---

## 1. The two stages

A transformer LLM generates text one token at a time. Every generation splits
into two very different phases:

### Stage 1 — Prefill (a.k.a. "prompt processing")
- The model runs **one forward pass over the entire prompt** at once.
- For PaliGemma the prompt = `image_tokens` (256 for the 224px model) + text tokens.
- All prompt positions are computed in parallel → this stage is **compute-bound**
  (big matmuls, high GPU utilization).
- It produces the **first output token**.
- Its latency is the **Time To First Token (TTFT)** — what a user perceives as
  "how long until it starts responding".

In the code, prefill is the **first** call to
`PaliGemmaForConditionalGeneration.forward(...)` in the generation loop, where
`kv_cache.num_items() == 0`. See `modeling_gemma.py`:
`_merge_input_ids_with_image_features` builds a full `[q_len, q_len]` mask in the
prefill branch.

### Stage 2 — Decode (a.k.a. "generation")
- The model feeds **one new token at a time**, reusing the **KV cache** for all
  previous tokens (so it does *not* recompute them).
- Each step does a tiny matmul (1 token) but must **read the whole KV cache** from
  memory → this stage is **memory-bandwidth-bound**, and GPU compute is
  underutilized.
- Its cost is measured as **Inter-Token Latency (ITL)** = seconds/token, and the
  aggregate **decode throughput** = tokens/sec.

In the code, decode is every subsequent `forward(...)` call, where `input_ids`
is a single token (`q_len == 1`) and `kv_cache.num_items() > 0`.

> **Key mental model:** Prefill is *parallel & compute-bound*. Decode is
> *sequential & memory-bound*. They have completely different performance
> characteristics, which is why production servers schedule them separately.

---

## 2. The KV cache

During attention, each token produces **Key** and **Value** vectors. Instead of
recomputing K/V for the whole sequence at every decode step, we cache them. This
is the `KVCache` class in `modeling_gemma.py`:

```python
self.key_cache[layer_idx]  = torch.cat([existing, new_key],   dim=-2)
self.value_cache[layer_idx]= torch.cat([existing, new_value], dim=-2)
```

### Memory formula
```
kv_bytes = 2 (K and V)
         * num_layers
         * batch_size
         * num_kv_heads       # Gemma uses Grouped-Query Attention -> fewer KV heads
         * seq_len            # prompt tokens + generated tokens
         * head_dim
         * bytes_per_element  # fp32=4, fp16/bf16=2
```

Two things to notice:
1. **KV cache grows linearly with sequence length.** A long prompt (or long
   generation) directly consumes more memory.
2. **GQA saves memory.** Gemma has `num_key_value_heads < num_attention_heads`,
   so the KV cache is much smaller than the number of query heads would suggest.

`benchmark_inference.py` reports both the **theoretical** size (from the formula)
and the **measured** size (summed from the real cache tensors) — they should match.

---

## 3. Metrics reference

| Stage | Metric | Meaning | Why it matters |
|-------|--------|---------|----------------|
| Load  | model load latency | time to read weights + build model | cold-start cost |
| Prefill | prompt tokens | image + text tokens fed | drives prefill cost & KV size |
| Prefill | prefill latency (TTFT) | time for the first token | user-perceived responsiveness |
| Prefill | prefill tokens/sec | prompt_tokens / prefill_latency | prompt-processing throughput |
| Decode | latency/token (ITL) | seconds per generated token | steady-state responsiveness |
| Decode | p50/p90/p99 ITL | tail latency | consistency under load |
| Decode | decode tokens/sec | generated / decode_time | generation throughput |
| Decode | total decode latency | all decode steps | bulk of long generations |
| KV     | cache tokens | prompt + generated tokens | memory driver |
| KV     | memory allocated | MB held by the cache | capacity / batch-size limit |
| E2E    | total latency | prefill + decode | overall wall clock |

---

## 4. Running the single-model benchmark

```bash
# edit paths in launch_benchmark.sh first (MODEL_PATH, IMAGE_FILE_PATH)
bash launch_benchmark.sh
```

Or directly, sweeping prompt lengths (word counts):

```bash
python benchmark_inference.py \
    --model_path "$HOME/projects/paligemma-weights/paligemma-3b-pt-224" \
    --image_file_path test_images/pic1.jpeg \
    --max_tokens_to_generate 100 \
    --sweep_prompt_lengths "1,8,32,64,128" \
    --json_out benchmark_results.json
```

### What to look for in the sweep
- **Prefill latency rises with prompt length** (more tokens to process) — roughly
  linear, then super-linear once attention's O(n²) term dominates for long prompts.
- **Decode latency/token also creeps up** as the prompt grows, because each decode
  step reads a larger KV cache (memory-bound).
- **KV-cache MB grows linearly** with `prompt_tokens + generated_tokens`.
- **TTFT vs. throughput trade-off**: long prompts hurt TTFT the most.

> Tip: because prefill dominates TTFT, note how 256 image tokens already impose a
> fixed floor on prefill cost even for a 1-word text prompt.

---

## 5. Stage 2 — Production serving with vLLM & concurrent users

This from-scratch loop processes **one request at a time** with a naive,
contiguous KV cache. Production servers do much more. The goal of Stage 2 is to
study how throughput and latency behave under **concurrent load**.

### 5.1 Why vLLM
- **PagedAttention**: stores the KV cache in fixed-size *blocks* (like OS virtual
  memory pages) instead of one contiguous buffer. This removes fragmentation and
  lets many sequences share GPU memory efficiently → far higher batch sizes.
- **Continuous batching**: new requests join the running batch as soon as slots
  free up, instead of waiting for the whole batch to finish. This keeps the GPU
  busy and massively improves throughput under concurrency.
- **Prefill/decode scheduling**: separates the compute-bound prefill from the
  memory-bound decode to maximize utilization.

### 5.2 Serve PaliGemma with vLLM
PaliGemma is supported by vLLM as a vision-language model.

```bash
pip install vllm

# Serve an OpenAI-compatible endpoint (multimodal)
vllm serve google/paligemma-3b-pt-224 \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.9
# endpoint: http://localhost:8000/v1
```

### 5.3 Concurrency study — the metrics that change under load
When you move from 1 user to N concurrent users, track:

| Metric | Single user | Under concurrency |
|--------|-------------|-------------------|
| TTFT | baseline | grows as prefills queue |
| ITL (p50/p99) | baseline | grows; tail latency spikes |
| **Throughput (system tok/s)** | low (GPU idle in decode) | **rises a lot** thanks to batching |
| GPU memory | one KV cache | many KV caches (PagedAttention) |
| Requests/sec | ~1/latency | scales until saturation |

The central trade-off: **continuous batching raises aggregate throughput at the
cost of higher per-request latency**. You want to find the concurrency level where
throughput is high but p99 latency is still acceptable.

### 5.4 Load-testing plan
Use the provided `stage2_vllm/load_test.py` (async client) to fire N concurrent
requests and measure per-request TTFT, ITL, and system throughput:

```bash
python stage2_vllm/load_test.py \
    --base_url http://localhost:8000/v1 \
    --model google/paligemma-3b-pt-224 \
    --image test_images/pic1.jpeg \
    --prompt "describe this image" \
    --concurrency 1,2,4,8,16,32 \
    --requests_per_level 32
```

Plot **throughput vs. concurrency** and **p99 latency vs. concurrency** to find
the knee of the curve (the saturation point).

### 5.5 Things to experiment with
- `--max-num-seqs` and `--gpu-memory-utilization` in `vllm serve` (batch capacity).
- Prompt length and `max_tokens` (prefill-heavy vs. decode-heavy workloads).
- `--quantization` (fp8/awq) to fit larger batches.
- Compare **your hand-written loop** (this repo, batch=1) vs **vLLM** at batch=1
  (isolates PagedAttention/kernel gains) and then vLLM under concurrency
  (isolates continuous-batching gains).

---

## 6. Suggested experiment write-up

1. **Baseline (this repo):** load time, TTFT, ITL, KV-cache MB for a fixed prompt.
2. **Prompt-length sweep (this repo):** show prefill & KV growth.
3. **vLLM single request:** same prompt — compare TTFT/ITL to the baseline.
4. **vLLM concurrency sweep:** throughput & p99 latency vs. concurrency.
5. **Conclusion:** where does batching help, and where does latency degrade?
