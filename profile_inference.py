"""
Performance profiling for PaliGemma inference — prefill vs decode.

This is the "LLM inference engineer" toolkit. It answers:
  - WHERE is time spent? (kernel-level breakdown via torch.profiler)
  - HOW MANY kernels are launched per token? (launch overhead signal)
  - Is each stage COMPUTE-bound or MEMORY-bound? (roofline / arithmetic intensity)
  - How does it scale with batch size and prompt length? (HBM + KV-cache pressure)

It profiles PREFILL and DECODE as separate regions because they have opposite
performance characteristics:
  - prefill: many tokens/forward -> high arithmetic intensity -> compute-bound
  - decode:  1 token/forward     -> low  arithmetic intensity -> memory-bound

Outputs
-------
  - console table: top CUDA kernels by self-time for each stage
  - kernel launch count per decode step (overhead indicator)
  - roofline estimate: arithmetic intensity vs the A40 ridge point
  - optional Chrome traces (--trace_dir) openable in chrome://tracing or Perfetto

Usage
-----
    python profile_inference.py \
        --model_path "$HOME/projects/paligemma-weights/paligemma-3b-pt-224" \
        --image_file_path test_images/pic1.jpeg \
        --prompt "caption en" \
        --batch_sizes "1,4,16" \
        --trace_dir traces
"""

from __future__ import annotations

import os
import time
from typing import List, Optional

import fire
import torch
from PIL import Image
from torch.profiler import profile, ProfilerActivity, record_function

from processing_paligemma import PaliGemmaProcessor
from modeling_gemma import KVCache, PaliGemmaForConditionalGeneration
from utils import load_hf_model


# --------------------------------------------------------------------------- #
# GPU spec table (extend as needed). Values are peak, vendor-published.
# --------------------------------------------------------------------------- #
GPU_SPECS = {
    # name substring : (fp32_tflops, bf16_tensor_tflops, hbm_GBps)
    "A40":  (37.4, 149.7, 696.0),
    "A100": (19.5, 312.0, 1555.0),
    "H100": (67.0, 989.0, 3350.0),
    "L40":  (90.5, 362.0, 864.0),
    "4090": (82.6, 165.0, 1008.0),
}


def _gpu_spec():
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    for key, spec in GPU_SPECS.items():
        if key in name:
            return name, spec
    return name, (37.4, 149.7, 696.0)  # default to A40-ish


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _bytes_of_params(model) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def _count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- #
# Profiling one stage
# --------------------------------------------------------------------------- #
def _profile_region(fn, active_steps: int, trace_path: Optional[str] = None):
    """Run `fn` under torch.profiler for `active_steps` iterations.

    Returns the profiler object so the caller can inspect key_averages().
    """
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(active_steps):
            fn()
        _sync()

    if trace_path:
        prof.export_chrome_trace(trace_path)
    return prof


def _top_kernels(prof, n=15) -> str:
    # Sort by self CUDA time if available, else self CPU time.
    try:
        tbl = prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=n
        )
    except Exception:
        tbl = prof.key_averages().table(
            sort_by="self_cpu_time_total", row_limit=n
        )
    return tbl


def _total_cuda_kernel_launches(prof) -> int:
    """Approximate number of CUDA kernel launches captured in the profile.

    Attribute names vary across torch versions, so we probe several. A kernel
    event is one that spent time on the device.
    """
    count = 0
    for evt in prof.key_averages():
        dev_time = (
            getattr(evt, "self_device_time_total", 0)
            or getattr(evt, "device_time_total", 0)
            or getattr(evt, "cuda_time_total", 0)
        )
        if dev_time and evt.count:
            count += evt.count
    return count


def _top_kernel_name(prof):
    """Name of the real GPU kernel with the most device time (skips wrappers)."""
    skip_prefixes = ("aten::", "cuda", "void at::native::record")
    best_name, best_time = "(none)", 0
    for evt in prof.key_averages():
        dev_time = (getattr(evt, "self_device_time_total", 0)
                    or getattr(evt, "cuda_time_total", 0))
        if not dev_time:
            continue
        name = evt.key
        if name.startswith(skip_prefixes):
            continue
        if dev_time > best_time:
            best_time, best_name = dev_time, name
    return best_name, best_time


def _avg_latency_ms(fn, iters: int) -> float:
    """Wall-clock average latency per call (with CUDA sync)."""
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - t0) / iters * 1e3


def _save_stage_report(path, title, roofline_text, prof, stage_latency_ms,
                       n_runs, extra_lines=None):
    """Write a neat, self-contained per-stage profile report."""
    top_name, top_time_us = _top_kernel_name(prof)
    launches = _total_cuda_kernel_launches(prof)
    try:
        table = prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=15)
    except Exception:
        table = prof.key_averages().table(
            sort_by="self_cpu_time_total", row_limit=15)

    with open(path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(title + "\n")
        f.write("=" * 70 + "\n\n")
        f.write(roofline_text + "\n\n")
        f.write(f"MEASURED LATENCY        : {stage_latency_ms:.2f} ms per call\n")
        f.write(f"HOTSPOT GPU KERNEL      : {top_name}\n")
        f.write(f"TOTAL GPU KERNEL LAUNCHES (over {n_runs} runs): {launches}\n")
        f.write(f"GPU KERNEL LAUNCHES PER CALL          : {launches / n_runs:.0f}\n")
        if extra_lines:
            f.write("\n" + "\n".join(extra_lines) + "\n")
        f.write("\nFULL KERNEL TABLE (sorted by GPU self-time):\n")
        f.write(table + "\n")

    print(f"  saved: {path}")
    print(f"    latency {stage_latency_ms:.2f} ms | {launches / n_runs:.0f} launches/call | hotspot {top_name[:50]}")


def _count_launches_of(fn, n):
    """Profile `fn` n times and return total GPU kernel launches (for attribution)."""
    prof = _profile_region(fn, active_steps=n)
    return _total_cuda_kernel_launches(prof)


# --------------------------------------------------------------------------- #
# Roofline estimate
# --------------------------------------------------------------------------- #
def roofline_report(model, batch_size, seq_len_prefill, kv_len_decode):
    """Rough FLOPs / bytes estimate to classify compute vs memory bound.

    We approximate per-forward FLOPs ~= 2 * params * num_query_tokens
    (each parameter participates in ~1 multiply-add per token), and bytes moved
    ~= params * dtype_bytes (weights read once per forward) + KV-cache traffic.
    These are order-of-magnitude estimates — the profiler timings are ground truth.

    Returns the report as a string (also printed by the caller).
    """
    name, (fp32_tflops, bf16_tflops, hbm_GBps) = _gpu_spec()
    params = _count_params(model)
    pbytes = _bytes_of_params(model)
    ridge_fp32 = (fp32_tflops * 1e12) / (hbm_GBps * 1e9)  # FLOP/byte

    def classify(ai):
        return "COMPUTE-bound" if ai > ridge_fp32 else "MEMORY-bound"

    # Prefill: num query tokens = seq_len_prefill
    flops_prefill = 2 * params * seq_len_prefill * batch_size
    bytes_prefill = pbytes  # weights read once (activations/KV smaller, ignore)
    ai_prefill = flops_prefill / bytes_prefill

    # Decode: num query tokens = 1 per sequence
    flops_decode = 2 * params * 1 * batch_size
    bytes_decode = pbytes  # still must read ALL weights for 1 token
    ai_decode = flops_decode / bytes_decode
    decode_floor_ms = (pbytes / (hbm_GBps * 1e9)) * 1e3

    lines = [
        f"=== ROOFLINE ({name}) ===",
        f"  params               : {params/1e9:.2f} B  ({pbytes/1e9:.2f} GB in fp32)",
        f"  peak fp32            : {fp32_tflops:.1f} TFLOP/s",
        f"  HBM bandwidth        : {hbm_GBps:.0f} GB/s",
        f"  ridge point (fp32)   : {ridge_fp32:.1f} FLOP/byte  (AI above this = compute-bound)",
        f"  --- batch_size={batch_size} ---",
        f"  PREFILL  seq_len={seq_len_prefill:<4}  AI={ai_prefill:10.1f} FLOP/byte  -> {classify(ai_prefill)}",
        f"  DECODE   (1 token)          AI={ai_decode:10.1f} FLOP/byte  -> {classify(ai_decode)}",
        f"  DECODE memory-bound floor : {decode_floor_ms:.2f} ms/token "
        f"(= read {pbytes/1e9:.1f} GB weights / {hbm_GBps:.0f} GB/s)",
    ]
    text = "\n".join(lines)
    print("\n" + text)
    return text


# --------------------------------------------------------------------------- #
# Build the two closures (prefill step, decode step) for a given batch size
# --------------------------------------------------------------------------- #
def _make_steps(model, processor, device, prompt, image, batch_size):
    inputs = processor(text=[prompt], images=[image])
    input_ids = inputs["input_ids"].to(device).repeat(batch_size, 1)
    attention_mask = inputs["attention_mask"].to(device).repeat(batch_size, 1)
    pixel_values = inputs["pixel_values"].to(device).repeat(batch_size, 1, 1, 1)
    seq_len = input_ids.shape[1]

    def prefill_once():
        kv = KVCache()
        with torch.no_grad():
            model(input_ids=input_ids, pixel_values=pixel_values,
                  attention_mask=attention_mask, kv_cache=kv)
        return kv

    # Pre-build a populated KV cache so decode steps are realistic.
    kv_cache = prefill_once()
    out_logits_tok = torch.argmax(
        torch.zeros(batch_size, 1, model.config.vocab_size, device=device), dim=-1
    )  # placeholder; replaced below
    # Do one real prefill to get a valid next token + grown mask
    kv_cache = KVCache()
    with torch.no_grad():
        o = model(input_ids=input_ids, pixel_values=pixel_values,
                  attention_mask=attention_mask, kv_cache=kv_cache)
    next_token = torch.argmax(o["logits"][:, -1, :], dim=-1, keepdim=True)
    dec_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), device=device)], dim=-1)

    state = {"next_token": next_token, "mask": dec_mask, "kv": kv_cache}

    def decode_once():
        with torch.no_grad():
            o = model(input_ids=state["next_token"], pixel_values=pixel_values,
                      attention_mask=state["mask"], kv_cache=state["kv"])
        state["next_token"] = torch.argmax(o["logits"][:, -1, :], dim=-1, keepdim=True)
        state["mask"] = torch.cat(
            [state["mask"], torch.ones((batch_size, 1), device=device)], dim=-1
        )

    return prefill_once, decode_once, seq_len


def main(
    model_path: str = None,
    image_file_path: str = None,
    prompt: str = "caption en",
    batch_sizes: str = "1,4,16",
    warmup_steps: int = 3,
    active_steps: int = 10,
    trace_dir: str = None,
    out_dir: str = "profile_report",
    only_cpu: bool = False,
):
    device = "cuda" if (not only_cpu and torch.cuda.is_available()) else "cpu"
    print("Device:", device, "|", torch.cuda.get_device_name(0) if device == "cuda" else "")

    print("Loading model ...")
    model, tokenizer = load_hf_model(model_path, device)
    model = model.to(device).eval()
    processor = PaliGemmaProcessor(
        tokenizer,
        model.config.vision_config.num_image_tokens,
        model.config.vision_config.image_size,
    )
    image = Image.open(image_file_path).convert("RGB")

    if isinstance(batch_sizes, (list, tuple)):
        sizes = [int(x) for x in batch_sizes]
    else:
        sizes = [int(x) for x in str(batch_sizes).split(",") if str(x).strip()]

    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    for b in sizes:
        print("\n" + "#" * 70)
        print(f"# BATCH SIZE = {b}")
        print("#" * 70)
        prefill_once, decode_once, seq_len = _make_steps(
            model, processor, device, prompt, image, b
        )

        # A vision-tower-only closure, to attribute how many decode kernels are
        # the redundant image re-encoding.
        inputs = processor(text=[prompt], images=[image])
        pv = inputs["pixel_values"].to(device).repeat(b, 1, 1, 1)

        def vision_only():
            with torch.no_grad():
                feat = model.vision_tower(pv.to(next(model.parameters()).dtype))
                model.multi_modal_projector(feat)

        # ---- Warmup ----
        for _ in range(warmup_steps):
            prefill_once()
            decode_once()
        _sync()

        # ---- Roofline classification (string, reused in the report files) ----
        roofline_text = roofline_report(model, b, seq_len_prefill=seq_len,
                                        kv_len_decode=seq_len + 1)

        # ---- PREFILL ----
        print("\n--- PREFILL ---")
        pre_runs = max(3, active_steps // 3)
        pre_latency = _avg_latency_ms(prefill_once, pre_runs)
        p_trace = os.path.join(trace_dir, f"prefill_b{b}.json") if trace_dir else None
        prof_pre = _profile_region(prefill_once, active_steps=pre_runs, trace_path=p_trace)
        _save_stage_report(
            os.path.join(out_dir, f"prefill_b{b}.txt"),
            f"PREFILL  (batch={b}, prompt_tokens={seq_len})  —  the COMPUTE-bound stage",
            roofline_text, prof_pre, pre_latency, pre_runs,
            extra_lines=[
                "WHAT TO SEE:",
                "  - hotspot is an *sgemm* (GEMM) kernel -> matrix x matrix -> compute-bound.",
                "  - AI (roofline above) sits far ABOVE the ridge point.",
                "  - this sets TTFT (time to first token).",
            ],
        )

        # ---- DECODE ----
        print("\n--- DECODE ---")
        dec_latency = _avg_latency_ms(decode_once, active_steps)
        d_trace = os.path.join(trace_dir, f"decode_b{b}.json") if trace_dir else None
        prof_dec = _profile_region(decode_once, active_steps=active_steps, trace_path=d_trace)

        # Attribution: how many decode kernels are the redundant vision tower?
        vis_launches = _count_launches_of(vision_only, active_steps) / active_steps
        total_dec_launches = _total_cuda_kernel_launches(prof_dec) / active_steps
        _save_stage_report(
            os.path.join(out_dir, f"decode_b{b}.txt"),
            f"DECODE  (batch={b}, 1 token/step)  —  the MEMORY-bound stage",
            roofline_text, prof_dec, dec_latency, active_steps,
            extra_lines=[
                "WHAT TO SEE:",
                "  - The LANGUAGE MODEL's own matmuls here are *gemv* kernels (matrix x",
                "    vector, 1 token) -> memory-bound. See gemv2T_kernel / gemvx in the table.",
                "  - BUT note the hotspot may be an *sgemm* (GEMM): that is the redundant",
                "    VISION tower (256 image patches = matrix) being re-run every decode step!",
                "    A GEMM showing up in 'decode' is itself the smoking gun for wasted work.",
                "  - AI (roofline above) for the real 1-token LM work sits far BELOW the ridge.",
                "  - this stage sets TPOT (time per output token).",
                "",
                "KERNEL-LAUNCH ATTRIBUTION (per decode step):",
                f"  total GPU kernels / step        : {total_dec_launches:.0f}",
                f"  redundant VISION tower / step   : {vis_launches:.0f}  "
                f"({100*vis_launches/max(total_dec_launches,1):.0f}% — wasted, image re-encoded!)",
                f"  language model + merge / step   : {total_dec_launches - vis_launches:.0f}",
                "",
                "FIX IDEAS (in priority order):",
                "  1. encode image ONCE (skip vision tower during decode)  -> removes the vision kernels",
                "  2. torch.compile the LM                                 -> fuses the elementwise chains",
                "  3. CUDA graphs / bf16                                   -> cut launch overhead + bytes",
            ],
        )

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

    print("\n" + "=" * 70)
    print(f"Neat reports written to {out_dir}/  (one .txt per stage per batch)")
    if trace_dir:
        print(f"Chrome traces written to {trace_dir}/  (open in ui.perfetto.dev)")
    print("=" * 70)


if __name__ == "__main__":
    fire.Fire(main)
