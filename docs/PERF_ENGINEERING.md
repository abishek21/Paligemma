# Performance Engineering — Prefill vs Decode on PaliGemma (A40)

How an LLM-inference engineer reasons about and *measures* where time goes. This
doc pairs with `profile_inference.py` (kernel profiler + roofline) and
`batch_study.py` (throughput vs batch size).

---

## 1. The two latency metrics: TTFT & TPOT

| Metric | Means | Dominated by | What a user feels |
|--------|-------|--------------|-------------------|
| **TTFT** (Time To First Token) | request → first token | **prefill** | responsiveness |
| **TPOT** (Time Per Output Token) = ITL | steady-state per-token time | **decode** | "typing speed" |

Total latency = `TTFT + (num_output_tokens − 1) × TPOT`.

Measured here (batch 1): **TTFT ≈ 0.12 s**, **TPOT ≈ 39 ms**.

---

## 1b. Throughput formulas (ms/step → tokens/sec)

A **decode step** produces **1 token per sequence**, so a batch of **B** sequences
yields **B tokens** per step. `ms/step` (the decode step time) **is TPOT**.

```
step_s       = ms_per_step / 1000

per_seq_tps  = 1000 / ms_per_step            # one USER's rate — NO batch term
system_tps   = batch_size * 1000 / ms_per_step   # whole GPU — × batch
             = batch_size * per_seq_tps

TPOT         = ms_per_step                   # the raw step time
```

Relationships:
```
system_tps  = batch_size × per_seq_tps
per_seq_tps = system_tps / batch_size
```

Worked example (batch 16, ms/step = 277):
```
per_seq_tps = 1000 / 277      = 3.6 tok/s    (one user)
system_tps  = 16 * 1000 / 277 = 57.8 tok/s   (whole GPU)   check: 16 × 3.6 ✓
```

**Memory aid:** per_seq ignores batch (one user); system multiplies by batch (all users).

### Speedup & efficiency (from `batch_study.py`)
```
speedup    = system_tps(B) / system_tps(1)   # total-throughput multiplier vs batch 1
efficiency = speedup / batch_size            # 1.0 (100%) = perfect linear scaling
```
As batch grows: **per_seq_tps ↓** (each user waits longer) while **system_tps ↑**
(more total work). That trade — helping the *server* at the cost of the *user* — is
the core of serving. Efficiency collapsing fast (here 100%→4% by batch 64) signals
the implementation hit the compute wall / wasted work early → optimization headroom
(encode-image-once, bf16, kernel fusion).

### Batch-sizing as capacity planning
The batch sweep is "batch-sizing 101": it finds (1) the throughput **plateau**,
(2) the **OOM ceiling**, and (3) the **latency/throughput trade-off**. You pick a
batch against an **SLA**, e.g. "per_seq_tps ≥ 10" → from the table above, even
batch 8 (6.5 tok/s/user) fails that SLA. Caveats: this is **static** batching on an
**unoptimized fp32** model — production uses **continuous batching** (vLLM) which
removes these artificial ceilings. Same concepts, higher limits.

---

## 2. Roofline model — the one mental tool that explains everything

**Arithmetic Intensity (AI)** = FLOPs performed ÷ bytes moved from HBM.

The GPU has two ceilings:
- **Compute ceiling**: peak FLOP/s (A40 fp32 ≈ 37.4 TFLOP/s)
- **Memory ceiling**: HBM bandwidth (A40 ≈ 696 GB/s)

**Ridge point** = compute ÷ bandwidth = `37.4e12 / 696e9 ≈ 53.7 FLOP/byte`.
- AI **above** ridge → **compute-bound** (good GPU utilization)
- AI **below** ridge → **memory-bound** (GPU starved, waiting on HBM)

### Measured AI on this model (2.92 B params, 11.69 GB fp32)

| Stage | batch | AI (FLOP/byte) | vs ridge 53.7 | Verdict |
|-------|------:|---------------:|--------------|---------|
| PREFILL | 1 | **130** | above | COMPUTE-bound |
| PREFILL | 16 | **2080** | far above | COMPUTE-bound |
| DECODE | 1 | **0.5** | far below | MEMORY-bound |
| DECODE | 16 | **8.0** | below | MEMORY-bound |

This is the whole story in one table.

---

## 3. WHY prefill is compute-bound

Prefill processes **all N prompt tokens in one pass**. Each weight is read from
HBM **once** but reused across **N** token-rows of matmul:

```
AI_prefill ∝ N   (hundreds)
```

A matmul `[N,d]×[d,d]` does `2·N·d²` FLOPs reading only ~`d²` weights → tons of
math per byte → saturates the compute units. At N=260, AI=130 > 53.7 → compute-bound.
At batch 16 the effective token count is 16×260 → AI=2080, deeply compute-bound.

**Profiler proof (prefill):** the hot kernels are **matrix–matrix GEMMs**:
```
ampere_sgemm_128x64_tn   63.6% of CUDA time   (tiled dense GEMM)
ampere_sgemm_128x128_tn  21.4%
```
`ampere_sgemm_*` = cuBLAS tiled matrix-matrix multiply = the compute-bound workhorse.

---

## 4. WHY decode (TPOT) is memory-bound

Decode processes **1 token per sequence**. You still must read **all 11.69 GB of
weights** from HBM, but to do math for just one token-row:

```
AI_decode ∝ batch   (≈0.5 at B=1, 8 at B=16)
```

Tiny AI → the GPU reads a mountain of weights to do a sliver of math → stalls on
memory → compute units idle → **memory-bound**.

**Memory-bound latency floor:**
```
TPOT_floor ≈ model_bytes / HBM_bandwidth = 11.69 GB / 696 GB/s ≈ 16.8 ms/token
```
We measured **~39 ms**. The gap (39 − 16.8 ≈ 22 ms) is overhead — the next section.

**Profiler proof (decode):** the hot kernels switch to **matrix–VECTOR GEMVs**:
```
gemv2T_kernel_val        22.8% of CUDA time
internal::gemvx...       10.9% + 9.8% + 3.1%
```
`gemv*` = matrix×vector = the classic memory-bound op. **Seeing GEMV instead of
GEMM is the signature of the decode regime.** Same model, opposite kernel mix —
purely because prefill feeds a matrix (N tokens) and decode feeds a vector (1 token).

---

## 5. Kernel-launch overhead — the second decode tax

The profiler counts **~3,682 CUDA kernel launches per *single* decode step** at
batch 1 (and ~6,598 at batch 16). That's enormous for producing one token.

Why so many? This from-scratch model is **unfused**: every `RMSNorm`, `mul`,
`add`, RoPE `rotate_half`, elementwise op, and the re-run **vision tower** each
launch separate tiny kernels. In the memory-bound decode regime, each kernel:
- has fixed **launch latency** (~microseconds of CPU→GPU dispatch), and
- re-reads/writes its activations to HBM (no fusion = extra HBM traffic).

With thousands of tiny kernels per token, **launch overhead + unfused HBM
round-trips** explain most of the 22 ms gap above the 16.8 ms floor.

> Signs you're launch-bound (visible in a Chrome trace): the GPU timeline has
> gaps between many short kernels, and CPU time ≈ GPU time (here Self CPU 327 ms
> ≈ Self CUDA 375 ms for decode — the CPU is busy *launching*, not computing).

---

## 6. What the optimizations target (and why)

| Technique | Fixes | Expected effect on decode |
|-----------|-------|---------------------------|
| **bf16/fp16 weights** | halves `model_bytes` | ~2× lower TPOT floor (16.8 → ~8.4 ms) |
| **torch.compile** | fuses elementwise ops, fewer kernels, CUDA graphs | cuts launch overhead + HBM round-trips |
| **CUDA graphs** | eliminates per-kernel launch latency | big win when launch-bound |
| **FlashAttention / fused attn** | fuses QKᵀ·softmax·V, no big score matrix in HBM | less HBM traffic in attention |
| **Encode image once** | stop re-running SigLIP every decode step | removes wasted vision kernels |
| **Batching** | raises AI (∝ batch) | more tokens per weight-read → higher throughput |
| **PagedAttention (vLLM)** | efficient KV cache, enables big batches | sustained high throughput |
| **Triton custom kernels** | hand-fused ops cuBLAS doesn't cover | targeted memory-traffic cuts |

Key insight: **PyTorch's `aten::mm` already dispatches to cuBLAS `ampere_sgemm`
/ `gemv`, which are near-optimal for a *single* matmul.** You rarely beat cuBLAS
on one GEMM. The wins come from **reducing the number of kernels, the precision,
and the HBM traffic *between* the matmuls** — i.e. fusion, graphs, lower precision,
and attention kernels. That's where torch.compile and Triton pay off.

---

## 7. KV-cache pressure (the batch × context study)

- KV-cache bytes = `2 · layers · batch · kv_heads · seq_len · head_dim · dtype_bytes`.
- This model: **1 KV head** (multi-query) → tiny cache (~16 MB at 459 tokens).
- But cache is **read every decode step**, adding to HBM traffic. As `batch` and
  `seq_len` grow, KV reads compete with weight reads for the 696 GB/s budget.
- Use `batch_study.py` (throughput vs batch) + this profiler to watch AI climb
  with batch and the regime shift from launch-bound → bandwidth-bound → compute-bound.

---

## 8. How to run the measurements

```bash
# Kernel profile + roofline (prefill vs decode), multiple batch sizes:
python profile_inference.py \
    --model_path "$HOME/projects/paligemma-weights/paligemma-3b-pt-224" \
    --image_file_path test_images/pic1.jpeg \
    --prompt "caption en" \
    --batch_sizes "1,4,16" \
    --trace_dir traces        # open traces/*.json in chrome://tracing or Perfetto

# Throughput vs batch size (GPU utilization curve):
python batch_study.py --batch_sizes "1,2,4,8,16,32,64" ...
```

### What to look for
1. **Roofline table**: prefill AI ≫ ridge (compute-bound); decode AI ≪ ridge (memory-bound).
2. **Kernel names**: `ampere_sgemm_*` (GEMM) in prefill vs `gemv*` in decode.
3. **Launches per decode step**: thousands → launch-overhead headroom for fusion/graphs.
4. **CPU≈CUDA time in decode**: CPU busy launching = launch-bound symptom.
5. **Chrome trace**: gaps between many short kernels on the GPU timeline.

---

## 9. One-paragraph summary

Prefill and decode are two different machines sharing weights. **Prefill** feeds a
*matrix* of tokens → high arithmetic intensity → cuBLAS **GEMM** kernels →
**compute-bound** (sets TTFT). **Decode** feeds a *vector* (one token) → low
arithmetic intensity → **GEMV** kernels → **memory-bound**, with its latency floored
by `model_bytes / HBM_bandwidth` (~16.8 ms here) and inflated by thousands of
unfused kernel launches per token (~39 ms actual). You don't beat cuBLAS on a
single matmul; you win decode by **lowering precision, fusing ops (torch.compile /
Triton), using CUDA graphs, cutting KV/attention HBM traffic, and batching** to
raise arithmetic intensity — which is exactly what vLLM productionizes.
