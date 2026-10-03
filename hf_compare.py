"""
Compare OUR from-scratch PaliGemma implementation vs HuggingFace transformers.

Both load the SAME weights and run the SAME prompt+image. We measure with the
SAME methodology as hbm_kv_study.py (manual prefill/decode split) so the numbers
are directly comparable:

    - TTFT   (prefill latency, time to first token)
    - TPOT   (per-token decode latency)
    - system throughput (tokens/sec across the batch)

We compare several configs for each implementation:
    ours:  fp32 | bf16 | bf16+compile
    hf:    fp32 | bf16 | bf16+compile

This tells us whether our hand-written model is competitive with HF's, and which
optimizations HF already has built in.

Usage
-----
    python hf_compare.py \
        --model_path "$HOME/projects/paligemma-weights/paligemma-3b-pt-224" \
        --image_file_path test_images/pic1.jpeg \
        --batch_size 32 --decode_steps 16
"""

from __future__ import annotations

import time
import fire
import torch
from PIL import Image


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _peak_bw():
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    peak = {"A40": 696.0, "A100": 1555.0, "H100": 3350.0}.get(
        next((k for k in ("A40", "A100", "H100") if k in name), ""), 696.0)
    return name, peak


# --------------------------------------------------------------------------- #
# OURS
# --------------------------------------------------------------------------- #
@torch.no_grad()
def bench_ours(model_path, image, prompt, device, batch_size, decode_steps,
               dtype, compile_lm, sdpa=False):
    from processing_paligemma import PaliGemmaProcessor
    from modeling_gemma import KVCache
    import modeling_gemma as mg
    from utils import load_hf_model

    mg.set_use_sdpa(sdpa)  # toggle fused FlashAttention path

    model, tok = load_hf_model(model_path, device)
    model = model.to(device).eval()
    if dtype != torch.float32:
        model = model.to(dtype)
    if compile_lm:
        model.language_model = torch.compile(model.language_model, dynamic=True)

    proc = PaliGemmaProcessor(tok, model.config.vision_config.num_image_tokens,
                              model.config.vision_config.image_size)
    ins = proc(text=[prompt], images=[image])
    input_ids = ins["input_ids"].to(device).repeat(batch_size, 1)
    attn = ins["attention_mask"].to(device).repeat(batch_size, 1)
    pv = ins["pixel_values"].to(device).repeat(batch_size, 1, 1, 1)

    def step(kv, iid, am):
        return model(input_ids=iid, pixel_values=pv, attention_mask=am, kv_cache=kv)

    # Warmup (compile / cudnn autotune)
    wk = KVCache()
    o = step(wk, input_ids, attn)
    nt = torch.argmax(o["logits"][:, -1, :], dim=-1, keepdim=True)
    am = torch.cat([attn, torch.ones((batch_size, 1), device=device)], dim=-1)
    for _ in range(3):
        o = step(wk, nt, am); nt = torch.argmax(o["logits"][:, -1, :], dim=-1, keepdim=True)
        am = torch.cat([am, torch.ones((batch_size, 1), device=device)], dim=-1)
    _sync()

    # PREFILL
    kv = KVCache()
    _sync(); t0 = time.perf_counter()
    o = step(kv, input_ids, attn)
    _sync(); ttft = time.perf_counter() - t0
    nt = torch.argmax(o["logits"][:, -1, :], dim=-1, keepdim=True)
    am = torch.cat([attn, torch.ones((batch_size, 1), device=device)], dim=-1)

    # DECODE
    times = []
    for _ in range(decode_steps):
        _sync(); s = time.perf_counter()
        o = step(kv, nt, am)
        _sync(); times.append(time.perf_counter() - s)
        nt = torch.argmax(o["logits"][:, -1, :], dim=-1, keepdim=True)
        am = torch.cat([am, torch.ones((batch_size, 1), device=device)], dim=-1)

    tpot = sum(times) / len(times)
    del model
    torch.cuda.empty_cache()
    return ttft, tpot


# --------------------------------------------------------------------------- #
# HUGGINGFACE
# --------------------------------------------------------------------------- #
@torch.no_grad()
def bench_hf(model_path, image, prompt, device, batch_size, decode_steps,
             dtype, compile_lm):
    from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
    from transformers.cache_utils import DynamicCache

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=dtype).to(device).eval()
    if compile_lm:
        model.language_model = torch.compile(model.language_model, dynamic=True)
    proc = AutoProcessor.from_pretrained(model_path)

    ins = proc(text=prompt, images=image, return_tensors="pt").to(device)
    input_ids = ins["input_ids"].repeat(batch_size, 1)
    attn = ins["attention_mask"].repeat(batch_size, 1)
    pv = ins["pixel_values"].repeat(batch_size, 1, 1, 1).to(dtype)

    def prefill(cache):
        return model(input_ids=input_ids, pixel_values=pv, attention_mask=attn,
                     past_key_values=cache, use_cache=True)

    def decode(cache, iid, am, cache_pos):
        return model(input_ids=iid, attention_mask=am, past_key_values=cache,
                     use_cache=True, cache_position=cache_pos)

    # Warmup
    wc = DynamicCache()
    o = prefill(wc)
    nt = torch.argmax(o.logits[:, -1, :], dim=-1, keepdim=True)
    am = torch.cat([attn, torch.ones((batch_size, 1), device=device, dtype=attn.dtype)], dim=-1)
    pos = torch.tensor([input_ids.shape[1]], device=device)
    for _ in range(3):
        o = decode(wc, nt, am, pos)
        nt = torch.argmax(o.logits[:, -1, :], dim=-1, keepdim=True)
        am = torch.cat([am, torch.ones((batch_size, 1), device=device, dtype=attn.dtype)], dim=-1)
        pos = pos + 1
    _sync()

    # PREFILL
    cache = DynamicCache()
    _sync(); t0 = time.perf_counter()
    o = prefill(cache)
    _sync(); ttft = time.perf_counter() - t0
    nt = torch.argmax(o.logits[:, -1, :], dim=-1, keepdim=True)
    am = torch.cat([attn, torch.ones((batch_size, 1), device=device, dtype=attn.dtype)], dim=-1)
    pos = torch.tensor([input_ids.shape[1]], device=device)

    # DECODE
    times = []
    for _ in range(decode_steps):
        _sync(); s = time.perf_counter()
        o = decode(cache, nt, am, pos)
        _sync(); times.append(time.perf_counter() - s)
        nt = torch.argmax(o.logits[:, -1, :], dim=-1, keepdim=True)
        am = torch.cat([am, torch.ones((batch_size, 1), device=device, dtype=attn.dtype)], dim=-1)
        pos = pos + 1

    tpot = sum(times) / len(times)
    del model
    torch.cuda.empty_cache()
    return ttft, tpot


def main(
    model_path: str = None,
    image_file_path: str = None,
    prompt: str = "caption en",
    batch_size: int = 32,
    decode_steps: int = 16,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name, peak = _peak_bw()
    image = Image.open(image_file_path).convert("RGB")
    print(f"Device: {device} | {name}\nBatch {batch_size}, decode_steps {decode_steps}, prompt {prompt!r}\n")

    # (label, dtype, compile, sdpa)   — sdpa only affects "ours"
    configs = [
        ("fp32",              torch.float32,  False, False),
        ("bf16",              torch.bfloat16, False, False),
        ("bf16+compile",      torch.bfloat16, True,  False),
        ("bf16+sdpa",         torch.bfloat16, False, True),
        ("bf16+sdpa+compile", torch.bfloat16, True,  True),
    ]

    header = f"{'impl / config':>26} {'TTFT_s':>8} {'TPOT_ms':>9} {'sys_tok/s':>10}"
    print(header); print("-" * len(header))
    rows = []
    for name_cfg, dt, comp, sdpa in configs:
        for impl, fn in [("ours", bench_ours), ("hf", bench_hf)]:
            # SDPA configs only apply to "ours"; HF already uses SDPA internally,
            # so skip the duplicate HF rows for the sdpa-specific labels.
            if "sdpa" in name_cfg and impl == "hf":
                continue
            try:
                if impl == "ours":
                    ttft, tpot = fn(model_path, image, prompt, device, batch_size,
                                    decode_steps, dt, comp, sdpa=sdpa)
                else:
                    ttft, tpot = fn(model_path, image, prompt, device, batch_size,
                                    decode_steps, dt, comp)
                sys_tps = batch_size / tpot
                print(f"{impl+' / '+name_cfg:>26} {ttft:>8.3f} {tpot*1e3:>9.2f} {sys_tps:>10.1f}")
                rows.append((impl, name_cfg, ttft, tpot, sys_tps))
            except Exception as e:  # noqa: BLE001
                print(f"{impl+' / '+name_cfg:>26}  ERROR: {str(e)[:60]}")

    print("\nNote: 'ours' uses the encode-image-once optimization (vision tower only")
    print("at prefill). HF handles the image-token merge in its own way. Both load the")
    print("same weights; greedy decode; identical prompt+image.")


if __name__ == "__main__":
    fire.Fire(main)
