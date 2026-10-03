"""
HBM-traffic & KV-cache-pressure benchmark at a fixed batch size.

Goal: at batch B (default 32), measure TTFT and TPOT, then SWEEP the context
length to watch the KV cache grow and show how it adds to HBM traffic — i.e.
simulate the "serving" conditions where KV-cache pressure actually bites.

How we measure HBM traffic (achieved-bandwidth method)
------------------------------------------------------
We can't read DRAM counters without Nsight Compute, so we MODEL the bytes moved
per decode step and divide by the measured step time:

    weight_bytes  = params * dtype_bytes                       # all weights, read once/step
    kv_read_bytes = 2 * layers * B * kv_heads * L * head_dim * dtype_bytes   # KV read every step
    total_bytes  ~= weight_bytes + kv_read_bytes
    achieved_BW   = total_bytes / step_time

Comparing achieved_BW to the GPU's peak HBM bandwidth tells us how memory-bound
we are. As context length L grows, kv_read_bytes grows and starts to rival the
weight read -> that is KV-cache bandwidth pressure.

Usage
-----
    python hbm_kv_study.py \
        --model_path "$HOME/projects/paligemma-weights/paligemma-3b-pt-224" \
        --image_file_path test_images/pic1.jpeg \
        --batch_size 32 \
        --context_lengths "260,512,1024"
"""

from __future__ import annotations

import time
from typing import List

import fire
import torch
from PIL import Image

from processing_paligemma import PaliGemmaProcessor
from modeling_gemma import KVCache, PaliGemmaForConditionalGeneration
from utils import load_hf_model


GPU_PEAK_BW = {"A40": 696.0, "A100": 1555.0, "H100": 3350.0, "L40": 864.0, "4090": 1008.0}


def _peak_bw():
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    for k, v in GPU_PEAK_BW.items():
        if k in name:
            return name, v
    return name, 696.0


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _params_and_bytes(model):
    p = sum(t.numel() for t in model.parameters())
    b = sum(t.numel() * t.element_size() for t in model.parameters())
    return p, b


def _build_prompt_ids(processor, image, device, text_tokens_target, batch_size):
    """Make a prompt whose total length ~= 256 image tokens + text_tokens_target."""
    # Repeat a word to hit the target text length (approx; tokenizer may differ).
    words = max(1, text_tokens_target)
    prompt = " ".join(["dog"] * words)
    inputs = processor(text=[prompt], images=[image])
    input_ids = inputs["input_ids"][:, : 256 + text_tokens_target].to(device)
    input_ids = input_ids.repeat(batch_size, 1)
    attn = torch.ones_like(input_ids, device=device)
    pv = inputs["pixel_values"].to(device).repeat(batch_size, 1, 1, 1)
    return input_ids, attn, pv


@torch.no_grad()
def measure(model, processor, image, device, batch_size, context_len, decode_steps,
            warmup: bool = False):
    text_target = max(1, context_len - 256)
    input_ids, attn, pv = _build_prompt_ids(processor, image, device, text_target, batch_size)
    seq_len = input_ids.shape[1]

    # Warmup: run a full prefill + a few decode steps so torch.compile does its
    # (slow) compilation here, NOT inside the timed region below.
    if warmup:
        wk = KVCache()
        wo = model(input_ids=input_ids, pixel_values=pv, attention_mask=attn, kv_cache=wk)
        wtok = torch.argmax(wo["logits"][:, -1, :], dim=-1, keepdim=True)
        wattn = torch.cat([attn, torch.ones((batch_size, 1), device=device)], dim=-1)
        for _ in range(3):
            wo = model(input_ids=wtok, pixel_values=pv, attention_mask=wattn, kv_cache=wk)
            wtok = torch.argmax(wo["logits"][:, -1, :], dim=-1, keepdim=True)
            wattn = torch.cat([wattn, torch.ones((batch_size, 1), device=device)], dim=-1)
        _sync()
        del wk

    kv = KVCache()
    # ---- PREFILL (TTFT) ----
    _sync(); t0 = time.perf_counter()
    o = model(input_ids=input_ids, pixel_values=pv, attention_mask=attn, kv_cache=kv)
    _sync(); ttft = time.perf_counter() - t0

    next_tok = torch.argmax(o["logits"][:, -1, :], dim=-1, keepdim=True)
    attn = torch.cat([attn, torch.ones((batch_size, 1), device=device)], dim=-1)

    # ---- DECODE (TPOT) ----
    step_times = []
    for _ in range(decode_steps):
        _sync(); s = time.perf_counter()
        o = model(input_ids=next_tok, pixel_values=pv, attention_mask=attn, kv_cache=kv)
        _sync(); step_times.append(time.perf_counter() - s)
        next_tok = torch.argmax(o["logits"][:, -1, :], dim=-1, keepdim=True)
        attn = torch.cat([attn, torch.ones((batch_size, 1), device=device)], dim=-1)

    tpot = sum(step_times) / len(step_times)
    kv_tokens = kv.num_items()
    # Measured KV cache bytes (sum of the real tensors)
    kv_bytes = 0
    for k, v in zip(kv.key_cache, kv.value_cache):
        kv_bytes += k.numel() * k.element_size() + v.numel() * v.element_size()
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0
    return {
        "seq_len": seq_len, "ttft": ttft, "tpot": tpot,
        "kv_tokens": kv_tokens, "kv_bytes": kv_bytes, "peak_mb": peak_mb,
    }


def main(
    model_path: str = None,
    image_file_path: str = None,
    batch_size: int = 32,
    context_lengths: str = "260,512,1024",
    decode_steps: int = 16,
    json_out: str = None,
    label: str = "baseline",
    compile: bool = False,
    compile_mode: str = "default",   # "default" | "reduce-overhead" | "max-autotune"
    only_cpu: bool = False,
):
    device = "cuda" if (not only_cpu and torch.cuda.is_available()) else "cpu"
    name, peak_bw = _peak_bw()
    print(f"Device: {device} | {name} | peak HBM ~{peak_bw:.0f} GB/s")

    print("Loading model ...")
    model, tok = load_hf_model(model_path, device)
    model = model.to(device).eval()

    if compile:
        # Compile the LANGUAGE MODEL (the per-token decode hot path). dynamic=True
        # tells the compiler to expect varying sequence lengths (the KV/context
        # grows every decode step) so it doesn't recompile on every token.
        print(f"Compiling language_model with torch.compile (mode={compile_mode}, dynamic=True) ...")
        kw = {} if compile_mode == "default" else {"mode": compile_mode}
        model.language_model = torch.compile(model.language_model, dynamic=True, **kw)

    processor = PaliGemmaProcessor(
        tok, model.config.vision_config.num_image_tokens,
        model.config.vision_config.image_size,
    )
    image = Image.open(image_file_path).convert("RGB")

    params, pbytes = _params_and_bytes(model)
    tc = model.config.text_config
    layers, kv_heads, head_dim = tc.num_hidden_layers, tc.num_key_value_heads, tc.head_dim
    dtype_bytes = next(model.parameters()).element_size()

    if isinstance(context_lengths, (list, tuple)):
        ctxs = [int(x) for x in context_lengths]
    else:
        ctxs = [int(x) for x in str(context_lengths).split(",") if str(x).strip()]

    print(f"\nModel: {params/1e9:.2f} B params, {pbytes/1e9:.2f} GB weights (fp32)")
    print(f"KV geometry: layers={layers}, kv_heads={kv_heads}, head_dim={head_dim}, "
          f"dtype_bytes={dtype_bytes}")
    print(f"Batch size: {batch_size}\n")

    header = (
        f"{'ctx_len':>8} {'ttft_s':>8} {'tpot_ms':>8} {'sys_tok/s':>10} "
        f"{'kv_MB':>8} {'wt_GB/stp':>10} {'kv_GB/stp':>10} {'ach_GB/s':>9} {'%peak':>6} {'peakMB':>8}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for L in ctxs:
        try:
            r = measure(model, processor, image, device, batch_size, L, decode_steps,
                        warmup=compile)
        except torch.cuda.OutOfMemoryError:
            print(f"{L:>8}  OOM — stopping")
            torch.cuda.empty_cache()
            break

        seq = r["seq_len"] + decode_steps  # approx final context
        # --- HBM traffic model (per decode step) ---
        weight_bytes = pbytes                                   # read all weights once/step
        kv_read_bytes = 2 * layers * batch_size * kv_heads * seq * head_dim * dtype_bytes
        total_bytes = weight_bytes + kv_read_bytes
        achieved_bw = total_bytes / r["tpot"]                   # bytes / s
        pct_peak = 100 * (achieved_bw / 1e9) / peak_bw
        sys_tps = batch_size / r["tpot"]

        print(
            f"{L:>8} {r['ttft']:>8.3f} {r['tpot']*1e3:>8.2f} {sys_tps:>10.1f} "
            f"{r['kv_bytes']/1e6:>8.1f} {weight_bytes/1e9:>10.2f} {kv_read_bytes/1e9:>10.3f} "
            f"{achieved_bw/1e9:>9.1f} {pct_peak:>5.0f}% {r['peak_mb']:>8.0f}"
        )
        rows.append({
            "context_len": L,
            "seq_len": r["seq_len"],
            "ttft_s": round(r["ttft"], 4),
            "tpot_ms": round(r["tpot"] * 1e3, 3),
            "system_tps": round(sys_tps, 2),
            "kv_cache_MB": round(r["kv_bytes"] / 1e6, 1),
            "weight_GB_per_step": round(weight_bytes / 1e9, 3),
            "kv_GB_per_step": round(kv_read_bytes / 1e9, 4),
            "achieved_GB_s": round(achieved_bw / 1e9, 2),
            "pct_peak_bw": round(pct_peak, 1),
            "peak_mem_MB": round(r["peak_mb"], 0),
        })
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

    if json_out:
        import json
        payload = {
            "label": label,
            "device": name,
            "peak_hbm_GB_s": peak_bw,
            "params_B": round(params / 1e9, 3),
            "weight_GB": round(pbytes / 1e9, 3),
            "batch_size": batch_size,
            "decode_steps": decode_steps,
            "kv_geometry": {"layers": layers, "kv_heads": kv_heads,
                            "head_dim": head_dim, "dtype_bytes": dtype_bytes},
            "rows": rows,
        }
        with open(json_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved results to {json_out}  (label='{label}')")

    print(
        "\nHow to read this:\n"
        "  - ttft_s  = prefill latency (TTFT) at this batch/context.\n"
        "  - tpot_ms = per-token decode latency (TPOT); rises as context grows.\n"
        "  - wt_GB/stp = weight bytes read per step (constant). kv_GB/stp = KV bytes\n"
        "    read per step (GROWS with context × batch). When kv_GB/stp approaches\n"
        "    wt_GB/stp, the KV cache is adding major HBM pressure.\n"
        "  - ach_GB/s / %peak = achieved HBM bandwidth vs the GPU peak. Near 100%\n"
        "    = fully memory-bandwidth-bound (the decode regime).\n"
        "  - kv_MB = actual KV cache memory held; peakMB = peak GPU memory."
    )


if __name__ == "__main__":
    fire.Fire(main)
