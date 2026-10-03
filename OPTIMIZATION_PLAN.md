# Optimization Plan — PaliGemma Inference (Stage 1.5)

Tomorrow's work: apply optimizations one at a time, re-measure, and compare to the
saved **baseline** (`results/hbm_baseline_b32.json`). Each optimization gets its own
before/after so we can attribute the gain.

> Measurement tools (already built):
> - `profile_inference.py` → kernel breakdown + roofline + launch attribution (writes `profile_report/`)
> - `batch_study.py` → throughput vs batch size
> - `hbm_kv_study.py` → TTFT/TPOT + HBM traffic + KV pressure (writes `results/*.json`)
> - `benchmark_inference.py` → prefill/decode/KV metrics (batch 1)

---

## Baseline to beat (batch 32, fp32, unoptimized)

| ctx | TTFT | TPOT | sys tok/s | achieved BW | %peak | peak mem |
|----:|-----:|-----:|----------:|------------:|------:|---------:|
| 260 | 3.15 s | 516.7 ms | 61.9 | 23.3 GB/s | **3.3%** | 20.2 GB |
| 512 | 5.38 s | 532.2 ms | 60.1 | 23.1 GB/s | **3.3%** | 28.4 GB |

Single-stream baseline (batch 1): **TTFT ≈ 0.12 s, TPOT ≈ 39 ms, ~25 tok/s**.
Batch sweep plateaued at **~62 tok/s**, OOM at batch 128.

**Key diagnosis:** only **3% of peak HBM bandwidth** → we are *overhead/waste-bound*,
not yet memory-bound. Decode fires **~3,682 kernels/token**, of which **~62% are the
redundant vision tower**. This is the headroom.

---

## RESULTS LOG (measured wins)

### ✅ Opt #1 — Encode image once  (DONE, Oct 3)
Guard the vision tower on the prefill condition; skip it during decode.
Output verified identical (`caption en` → same text). Measured at **batch 32**:

| metric (ctx 260) | baseline (fp32) | encode-once | change |
|------------------|----------------:|------------:|-------:|
| **TPOT** | 516.7 ms | **33.4 ms** | **15.5× faster** |
| **system tok/s** | 61.9 | **958.5** | **15.5×** |
| **achieved HBM BW** | 23.3 GB/s (3%) | **360 GB/s (52%)** | **15×** |

Batch-1 profiler: decode **38.75 → 19.74 ms**, kernels/token **3,682 → 2,048**,
decode hotspot flipped from `ampere_sgemm` (vision GEMM) → `gemv2T_kernel` (real LM).
Interpretation: removing the redundant image re-encode moved decode OUT of the
"waste-bound" corner (3% BW) INTO the genuine memory-bound regime (52% BW).
Files: `results/hbm_encode_once_b32.json`, `profile_report_encode_once/`.

Remaining headroom: still ~2,048 unfused kernels/token (→ torch.compile) and fp32
(→ bf16 halves the 16.8 ms memory floor).

---

## The optimizations, in priority order

### 1. Encode image ONCE  (biggest, easiest win)  ⭐  — ✅ DONE (see results log)
- **Problem:** `forward` re-runs the full SigLIP vision tower every decode step; the
  result is discarded (no `<image>` slots in decode). ~62% of decode kernels wasted.
- **Fix:** guard the vision tower on the prefill condition
  (`kv_cache is None or kv_cache.num_items() == 0`); skip it during decode.
  File: `modeling_gemma.py` `forward` (~line 530) + `_merge_input_ids_with_image_features`.
- **Expected:** decode kernels/token drops ~2,284; TPOT falls; achieved BW rises;
  batch ceiling increases (less activation memory).
- **Measure:** re-run `profile_inference.py` (watch vision kernels → 0) and
  `hbm_kv_study.py` (watch %peak jump).

### 2. torch.compile the language model
- **Problem:** ~1,704 unfused LM kernels/token (RMSNorm, RoPE, elementwise chains).
- **Fix:** `model.language_model = torch.compile(model.language_model)` (or compile
  the decoder layer). Handle warmup + static shapes (decode q_len=1 is static).
- **Expected:** fuses elementwise ops → fewer launches → lower TPOT.
- **Caveat:** dynamic context length may trigger recompiles; consider
  `mode="reduce-overhead"` (CUDA graphs) or padding to fixed buckets.
- **Measure:** kernel launches/token should drop sharply (like hello-world exp4: 311→3).

### 3. bf16 / fp16 weights
- **Problem:** fp32 = 11.69 GB to read per step; memory floor ~16.8 ms.
- **Fix:** load/cast model to bfloat16.
- **Expected:** ~½ the bytes → ~½ the memory-bound floor (→ ~8.4 ms); ~2× batch
  capacity (weights 11.7 GB → 5.85 GB).
- **Measure:** `hbm_kv_study.py` — weight_GB/step halves; TPOT floor halves; OOM ceiling rises.

### 4. CUDA graphs (mode="reduce-overhead")
- **Problem:** residual per-kernel launch latency after fusion.
- **Fix:** capture the decode step as a CUDA graph and replay.
- **Expected:** near-zero CPU dispatch overhead for decode.
- **Caveat:** requires static shapes / fixed KV layout — trickier with growing context.

### 5. (Optional) FlashAttention / fused attention
- **Problem:** naive attention materializes `[B, heads, q, kv]` score tensors in HBM.
- **Fix:** use `F.scaled_dot_product_attention` (flash backend) instead of manual QK/softmax/V.
- **Expected:** less attention HBM traffic, especially at long context.

### 6. (Optional) Triton custom kernel
- Hand-fuse something cuBLAS/Inductor don't cover (e.g. RMSNorm+RoPE) to learn Triton.
- Low priority; mostly educational.

---

## Experiment protocol (do this for each optimization)

1. Apply the change behind a flag or on a copy so baseline still runs.
2. Re-run the **same** commands with a new `--label` and `--json_out`:
   ```bash
   python hbm_kv_study.py ... --label "encode_once" --json_out results/hbm_encode_once_b32.json
   python profile_inference.py ... --out_dir profile_report_encode_once
   python batch_study.py ... (note new plateau + OOM ceiling)
   ```
3. Compare JSONs (TPOT, achieved BW %peak, kernels/token, OOM ceiling).
4. Record the win in a results table in `PERF_ENGINEERING.md`.

---

## Expected cumulative story (hypothesis to verify)

| stage | TPOT (b32, ctx260) | %peak BW | kernels/token | batch ceiling |
|-------|-------------------:|---------:|--------------:|--------------:|
| baseline (fp32) | 516 ms | 3% | 3,682 | ~64 |
| + encode once | ↓ (less compute) | ↑ | ~1,400 | ↑ |
| + torch.compile | ↓↓ | ↑↑ | ~hundreds | — |
| + bf16 | ↓ (½ floor) | — | — | ~2× |
| + CUDA graphs | ↓ (launch) | ↑ | — | — |

Goal: push achieved BW from 3% toward a healthy memory-bound regime, then halve the
floor with bf16 — turning the naive loop into something that approaches what vLLM
does, before we hand off to vLLM for continuous batching (Stage 2).

---

## Then: Stage 2 — vLLM
Once we've maxed the hand-written model, compare against `stage2_vllm/` (serve +
concurrency load test) to see PagedAttention + continuous batching blow past the
static-batch ceiling.
