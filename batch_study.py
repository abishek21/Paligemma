"""
Batch-size study — WHY single-stream inference wastes the GPU (the vLLM motivation).

The decode stage is *memory-bandwidth bound*: each step we load the full model
weights + KV cache from GPU memory just to compute ONE token. At batch size 1 the
GPU's compute units sit mostly idle waiting on memory.

Key idea: if we process B independent sequences at once (a batched decode), we
reuse the SAME weight read for all B sequences. So:

    - per-step latency stays *almost flat* as B grows (until compute-bound)
    - total throughput (tokens/sec across all sequences) scales *nearly linearly*

That gap between "flat latency" and "linear throughput" IS the spare GPU capacity
that a real serving engine (vLLM) harvests with continuous batching.

This script measures decode throughput vs. batch size using the actual model, by
duplicating one prompt B times and running a batched generation loop.

NOTE: the from-scratch model in this repo asserts `attention_mask == 1` (no
padding), so we keep all sequences the same length — which is exactly what we
want for a clean batch-scaling measurement.

Usage
-----
    python batch_study.py \
        --model_path "$HOME/projects/paligemma-weights/paligemma-3b-pt-224" \
        --image_file_path test_images/pic1.jpeg \
        --prompt "caption en" \
        --decode_steps 64 \
        --batch_sizes "1,2,4,8,16,32"
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


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _now() -> float:
    return time.perf_counter()


@torch.no_grad()
def measure_batch(
    model, processor, device, prompt, image, batch_size, decode_steps
) -> dict:
    """Run prefill + `decode_steps` batched decode steps for a given batch size.

    Returns per-step decode latency and aggregate throughput.
    """
    # Build inputs for a single sequence, then repeat along the batch dim.
    inputs = processor(text=[prompt], images=[image])
    input_ids = inputs["input_ids"].to(device).repeat(batch_size, 1)
    attention_mask = inputs["attention_mask"].to(device).repeat(batch_size, 1)
    pixel_values = inputs["pixel_values"].to(device).repeat(batch_size, 1, 1, 1)

    kv_cache = KVCache()

    # ---- Prefill ----
    _sync(device)
    t0 = _now()
    out = model(input_ids=input_ids, pixel_values=pixel_values,
                attention_mask=attention_mask, kv_cache=kv_cache)
    _sync(device)
    prefill_s = _now() - t0
    kv_cache = out["kv_cache"]

    next_token = torch.argmax(out["logits"][:, -1, :], dim=-1, keepdim=True)  # [B,1]
    attention_mask = torch.cat(
        [attention_mask, torch.ones((batch_size, 1), device=device)], dim=-1
    )

    # ---- Batched decode ----
    step_latencies: List[float] = []
    for _ in range(decode_steps):
        _sync(device)
        s = _now()
        out = model(input_ids=next_token, pixel_values=pixel_values,
                    attention_mask=attention_mask, kv_cache=kv_cache)
        _sync(device)
        step_latencies.append(_now() - s)

        kv_cache = out["kv_cache"]
        next_token = torch.argmax(out["logits"][:, -1, :], dim=-1, keepdim=True)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((batch_size, 1), device=device)], dim=-1
        )

    mean_step_s = sum(step_latencies) / len(step_latencies)
    # Each decode step produces `batch_size` tokens (one per sequence).
    system_tps = batch_size / mean_step_s
    per_seq_tps = 1.0 / mean_step_s

    peak_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2) if device == "cuda" else 0.0
    )

    return {
        "batch_size": batch_size,
        "prefill_s": prefill_s,
        "decode_ms_per_step": mean_step_s * 1e3,
        "per_seq_tokens_per_s": per_seq_tps,
        "system_tokens_per_s": system_tps,
        "peak_mb": peak_mb,
    }


def main(
    model_path: str = None,
    image_file_path: str = None,
    prompt: str = "caption en",
    decode_steps: int = 64,
    batch_sizes: str = "1,2,4,8,16,32",
    only_cpu: bool = False,
):
    device = "cpu"
    if not only_cpu and torch.cuda.is_available():
        device = "cuda"
    print("Device:", device)

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

    # Warmup (excludes lazy init from the first measured point).
    print("Warmup ...")
    measure_batch(model, processor, device, prompt, image, 1, 4)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    rows = []
    for b in sizes:
        try:
            r = measure_batch(model, processor, device, prompt, image, b, decode_steps)
            rows.append(r)
            print(f"  batch={b:3d}  done  "
                  f"({r['decode_ms_per_step']:.1f} ms/step, "
                  f"{r['system_tokens_per_s']:.1f} sys tok/s)")
        except torch.cuda.OutOfMemoryError:
            print(f"  batch={b:3d}  OOM — stopping sweep")
            torch.cuda.empty_cache()
            break
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

    # ---- Report ----
    print("\n\nBATCH-SIZE DECODE SCALING")
    header = (
        f"{'batch':>6} {'ms/step':>9} {'per_seq_tps':>12} {'system_tps':>11} "
        f"{'speedup':>8} {'efficiency':>10} {'peak_MB':>9}"
    )
    print(header)
    print("-" * len(header))
    base_tps = rows[0]["system_tokens_per_s"] if rows else 1.0
    for r in rows:
        speedup = r["system_tokens_per_s"] / base_tps
        efficiency = speedup / r["batch_size"]     # 1.0 == perfect linear scaling
        print(
            f"{r['batch_size']:>6} "
            f"{r['decode_ms_per_step']:>9.2f} "
            f"{r['per_seq_tokens_per_s']:>12.2f} "
            f"{r['system_tokens_per_s']:>11.2f} "
            f"{speedup:>7.2f}x "
            f"{efficiency*100:>9.1f}% "
            f"{r['peak_mb']:>9.1f}"
        )

    print(
        "\nHow to read this:\n"
        "  - ms/step barely rises while batch grows  -> GPU had spare compute at B=1.\n"
        "  - system_tps climbs ~linearly (speedup ~= batch) -> that's the wasted capacity\n"
        "    a serving engine reclaims by batching many users' decode steps together.\n"
        "  - When ms/step finally starts rising and efficiency drops, you've become\n"
        "    compute-bound: the practical max batch for your latency budget.\n"
        "  - per_seq_tps is what each *individual* user feels (stays ~flat until saturation)."
    )


if __name__ == "__main__":
    fire.Fire(main)
