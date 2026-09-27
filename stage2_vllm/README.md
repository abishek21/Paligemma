# Stage 2 — vLLM production serving & concurrency study

This folder extends the single-model benchmark (`../benchmark_inference.py`) to a
**production-grade serving** setup and studies behaviour under **concurrent users**.

See `../BENCHMARKING.md` section 5 for the full explanation of *why* vLLM changes
the performance picture (PagedAttention + continuous batching).

## Files
- `serve_vllm.sh`  — starts a vLLM OpenAI-compatible server for PaliGemma.
- `load_test.py`   — async client that drives concurrent load and reports TTFT,
                     ITL, system throughput, and p50/p99 latencies per concurrency level.

## Quick start

1. Install and serve (needs a CUDA GPU):
   ```bash
   pip install vllm openai
   bash stage2_vllm/serve_vllm.sh
   ```

2. In another terminal, run the concurrency sweep:
   ```bash
   pip install openai
   python stage2_vllm/load_test.py \
       --base_url http://localhost:8000/v1 \
       --model google/paligemma-3b-pt-224 \
       --image test_images/pic1.jpeg \
       --prompt "describe this image" \
       --max_tokens 64 \
       --concurrency 1,2,4,8,16,32 \
       --requests_per_level 32
   ```

## What you'll observe
- **System throughput (tok/s) rises** with concurrency thanks to continuous
  batching — until the GPU saturates.
- **TTFT and p99 latency climb** as requests queue for prefill.
- The **"knee"** of the curve is where throughput plateaus but tail latency keeps
  growing — the practical max concurrency for your latency budget.

## Experiments to try
- Vary `--max-num-seqs` and `--gpu-memory-utilization` in `serve_vllm.sh`.
- Compare prefill-heavy (long prompt, few `max_tokens`) vs decode-heavy
  (short prompt, many `max_tokens`) workloads.
- Add `--quantization fp8` (if supported) to fit larger batches.
- Compare against the hand-written batch=1 loop in the parent repo to isolate the
  gains from PagedAttention vs. continuous batching.
