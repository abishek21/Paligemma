"""
Instrumented inference for PaliGemma to study the two stages of LLM generation:

    1. PREFILL  -> process the whole prompt (image tokens + text tokens) in one
                   forward pass and produce the FIRST generated token.
    2. DECODE   -> generate the remaining tokens one at a time, each pass reusing
                   the KV cache so we only feed a single new token.

Metrics collected
------------------
Model load:
    - load latency (seconds)

Prefill:
    - number of prompt tokens (image tokens + text tokens)
    - prefill latency (seconds)  == time-to-first-token (TTFT)
    - prefill throughput (tokens/sec)

Decode:
    - per-token latency (seconds/token)
    - inter-token latency distribution (p50/p90/p99)
    - total decode latency
    - decode throughput (tokens/sec)

KV cache:
    - number of tokens held in the cache
    - theoretical + measured memory allocated by the cache

End to end:
    - total generation latency

You can sweep several prompt lengths in a single run to see how prefill cost
grows with the prompt (roughly linear for compute, and the KV cache grows
linearly in memory).

Usage
-----
    python benchmark_inference.py \
        --model_path "$HOME/projects/paligemma-weights/paligemma-3b-pt-224" \
        --image_file_path test_images/pic1.jpeg \
        --prompt "this building is" \
        --max_tokens_to_generate 100 \
        --sweep_prompt_lengths "1,8,32,64,128" \
        --only_cpu False
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import fire
import torch
from PIL import Image

from processing_paligemma import PaliGemmaProcessor
from modeling_gemma import KVCache, PaliGemmaForConditionalGeneration
from utils import load_hf_model


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _sync(device: str) -> None:
    """Make sure all queued GPU/MPS work is finished before we read the clock.

    On CPU this is a no-op. On CUDA/MPS the kernels are launched
    asynchronously, so without a sync our timers would measure launch time,
    not compute time.
    """
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _now() -> float:
    return time.perf_counter()


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _dtype_bytes(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass
class KVCacheStats:
    num_tokens: int = 0
    num_layers: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    dtype_bytes: int = 0
    theoretical_bytes: int = 0          # from config math
    measured_bytes: int = 0             # summed from the actual tensors

    @property
    def theoretical_mb(self) -> float:
        return self.theoretical_bytes / (1024 ** 2)

    @property
    def measured_mb(self) -> float:
        return self.measured_bytes / (1024 ** 2)


@dataclass
class RunResult:
    prompt_text: str = ""
    prompt_tokens: int = 0              # total tokens fed at prefill (image + text)
    text_tokens: int = 0               # just the text portion
    image_tokens: int = 0

    prefill_latency_s: float = 0.0     # == time to first token (TTFT)
    prefill_tokens_per_s: float = 0.0

    decode_tokens: int = 0
    decode_total_latency_s: float = 0.0
    decode_per_token_latencies_s: List[float] = field(default_factory=list)

    total_latency_s: float = 0.0
    generated_text: str = ""

    kv_cache: KVCacheStats = field(default_factory=KVCacheStats)

    # ---- derived metrics ----
    @property
    def decode_mean_latency_s(self) -> float:
        n = len(self.decode_per_token_latencies_s)
        return (sum(self.decode_per_token_latencies_s) / n) if n else 0.0

    @property
    def decode_tokens_per_s(self) -> float:
        return (self.decode_tokens / self.decode_total_latency_s) if self.decode_total_latency_s else 0.0

    def summary_dict(self) -> dict:
        lat = self.decode_per_token_latencies_s
        return {
            "prompt_tokens": self.prompt_tokens,
            "image_tokens": self.image_tokens,
            "text_tokens": self.text_tokens,
            "prefill_latency_s": round(self.prefill_latency_s, 4),
            "prefill_tokens_per_s": round(self.prefill_tokens_per_s, 2),
            "ttft_s": round(self.prefill_latency_s, 4),
            "decode_tokens": self.decode_tokens,
            "decode_mean_latency_ms": round(self.decode_mean_latency_s * 1e3, 2),
            "decode_p50_latency_ms": round(_percentile(lat, 50) * 1e3, 2),
            "decode_p90_latency_ms": round(_percentile(lat, 90) * 1e3, 2),
            "decode_p99_latency_ms": round(_percentile(lat, 99) * 1e3, 2),
            "decode_tokens_per_s": round(self.decode_tokens_per_s, 2),
            "decode_total_latency_s": round(self.decode_total_latency_s, 4),
            "total_latency_s": round(self.total_latency_s, 4),
            "kv_cache_tokens": self.kv_cache.num_tokens,
            "kv_cache_theoretical_mb": round(self.kv_cache.theoretical_mb, 2),
            "kv_cache_measured_mb": round(self.kv_cache.measured_mb, 2),
        }


# --------------------------------------------------------------------------- #
# KV cache accounting
# --------------------------------------------------------------------------- #
def compute_kv_cache_stats(model, kv_cache: KVCache) -> KVCacheStats:
    """Read the real KV-cache tensors and also compute the theoretical size.

    Memory model:
        bytes = 2 (key + value)
              * num_layers
              * batch_size
              * num_kv_heads
              * seq_len
              * head_dim
              * bytes_per_element
    """
    text_cfg = model.config.text_config
    num_layers = text_cfg.num_hidden_layers
    num_kv_heads = text_cfg.num_key_value_heads
    head_dim = text_cfg.head_dim

    num_tokens = kv_cache.num_items()

    # Measured: sum the actual allocated tensors in the cache.
    measured = 0
    dtype_bytes = 0
    batch_size = 1
    for k, v in zip(kv_cache.key_cache, kv_cache.value_cache):
        measured += k.numel() * k.element_size()
        measured += v.numel() * v.element_size()
        dtype_bytes = k.element_size()
        batch_size = k.shape[0]

    theoretical = (
        2 * num_layers * batch_size * num_kv_heads * num_tokens * head_dim * (dtype_bytes or 4)
    )

    return KVCacheStats(
        num_tokens=num_tokens,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype_bytes=dtype_bytes or 4,
        theoretical_bytes=theoretical,
        measured_bytes=measured,
    )


# --------------------------------------------------------------------------- #
# The instrumented generation loop
# --------------------------------------------------------------------------- #
def run_generation(
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    device: str,
    prompt: str,
    image: Image.Image,
    max_tokens_to_generate: int,
    do_sample: bool = False,
    temperature: float = 0.8,
    top_p: float = 0.9,
    ignore_eos: bool = False,
) -> RunResult:
    result = RunResult(prompt_text=prompt)

    model_inputs = processor(text=[prompt], images=[image])
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]

    result.prompt_tokens = int(input_ids.shape[1])
    result.image_tokens = int(processor.image_seq_length)
    result.text_tokens = result.prompt_tokens - result.image_tokens

    kv_cache = KVCache()
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens: List[torch.Tensor] = []

    total_start = _now()

    # ------------------------------------------------------------------ #
    # STAGE 1: PREFILL — one forward pass over the entire prompt.
    #          Produces the first generated token. Latency == TTFT.
    # ------------------------------------------------------------------ #
    _sync(device)
    prefill_start = _now()
    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        kv_cache=kv_cache,
    )
    _sync(device)
    result.prefill_latency_s = _now() - prefill_start

    kv_cache = outputs["kv_cache"]
    next_token_logits = outputs["logits"][:, -1, :]
    next_token = _pick_token(next_token_logits, do_sample, temperature, top_p)
    generated_tokens.append(next_token.squeeze(0))

    # Prefill throughput = prompt tokens processed / prefill time.
    result.prefill_tokens_per_s = (
        result.prompt_tokens / result.prefill_latency_s if result.prefill_latency_s else 0.0
    )

    # ------------------------------------------------------------------ #
    # STAGE 2: DECODE — feed one token at a time, reuse KV cache.
    # ------------------------------------------------------------------ #
    input_ids = next_token
    attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), device=device)], dim=-1
    )

    decode_start = _now()
    if ignore_eos or generated_tokens[-1].item() != stop_token:
        for _ in range(max_tokens_to_generate - 1):
            _sync(device)
            tok_start = _now()
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
            )
            _sync(device)
            result.decode_per_token_latencies_s.append(_now() - tok_start)

            kv_cache = outputs["kv_cache"]
            next_token_logits = outputs["logits"][:, -1, :]
            next_token = _pick_token(next_token_logits, do_sample, temperature, top_p)
            generated_tokens.append(next_token.squeeze(0))

            if not ignore_eos and next_token.item() == stop_token:
                break

            input_ids = next_token
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), device=device)], dim=-1
            )

    result.decode_total_latency_s = _now() - decode_start
    result.total_latency_s = _now() - total_start
    result.decode_tokens = len(result.decode_per_token_latencies_s)

    # KV cache stats measured at the end (holds prompt + all generated tokens).
    result.kv_cache = compute_kv_cache_stats(model, kv_cache)

    decoded = processor.tokenizer.decode(
        torch.cat(generated_tokens, dim=-1), skip_special_tokens=True
    )
    result.generated_text = decoded
    return result


def _pick_token(logits, do_sample, temperature, top_p):
    if do_sample:
        probs = torch.softmax(logits / temperature, dim=-1)
        return _sample_top_p(probs, top_p)
    return torch.argmax(logits, dim=-1, keepdim=True)


def _sample_top_p(probs: torch.Tensor, p: float):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)


# --------------------------------------------------------------------------- #
# Pretty printing
# --------------------------------------------------------------------------- #
def print_result(result: RunResult) -> None:
    s = result.summary_dict()
    print("\n" + "=" * 60)
    print(f"PROMPT: {result.prompt_text!r}")
    print("-" * 60)
    print(f"  prompt tokens        : {s['prompt_tokens']}  "
          f"(image={s['image_tokens']}, text={s['text_tokens']})")
    print("  PREFILL (stage 1)")
    print(f"    latency / TTFT     : {s['prefill_latency_s']:.4f} s")
    print(f"    throughput         : {s['prefill_tokens_per_s']:.2f} tok/s")
    print("  DECODE (stage 2)")
    print(f"    tokens generated   : {s['decode_tokens']}")
    print(f"    mean latency/token : {s['decode_mean_latency_ms']:.2f} ms")
    print(f"    p50 / p90 / p99    : "
          f"{s['decode_p50_latency_ms']:.2f} / "
          f"{s['decode_p90_latency_ms']:.2f} / "
          f"{s['decode_p99_latency_ms']:.2f} ms")
    print(f"    throughput         : {s['decode_tokens_per_s']:.2f} tok/s")
    print(f"    total decode time  : {s['decode_total_latency_s']:.4f} s")
    print("  KV CACHE")
    print(f"    tokens in cache    : {s['kv_cache_tokens']}")
    print(f"    memory (theory)    : {s['kv_cache_theoretical_mb']:.2f} MB")
    print(f"    memory (measured)  : {s['kv_cache_measured_mb']:.2f} MB")
    print("  END TO END")
    print(f"    total latency      : {s['total_latency_s']:.4f} s")
    print(f"  OUTPUT: {result.generated_text!r}")
    print("=" * 60)


def print_sweep_table(results: List[RunResult]) -> None:
    print("\n\nPROMPT-LENGTH SWEEP SUMMARY")
    header = (
        f"{'prompt_tok':>10} {'ttft_s':>8} {'prefill_tps':>12} "
        f"{'dec_ms/tok':>11} {'dec_tps':>8} {'kv_tokens':>10} {'kv_MB':>8} {'total_s':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        s = r.summary_dict()
        print(
            f"{s['prompt_tokens']:>10} "
            f"{s['prefill_latency_s']:>8.4f} "
            f"{s['prefill_tokens_per_s']:>12.2f} "
            f"{s['decode_mean_latency_ms']:>11.2f} "
            f"{s['decode_tokens_per_s']:>8.2f} "
            f"{s['kv_cache_tokens']:>10} "
            f"{s['kv_cache_measured_mb']:>8.2f} "
            f"{s['total_latency_s']:>8.4f}"
        )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
# A pool of coherent, image-relevant sentences (the test image is a dog).
# We stitch these together to build natural prompts of a target word length,
# instead of repeating a single word — so the text is realistic while we still
# control the length for the prefill / KV-cache scaling study.
_DOG_SENTENCES = [
    "Describe this dog in detail.",
    "The dog is sitting outdoors on a bright sunny day.",
    "Note the color of its fur, the shape of its ears, and its expression.",
    "Explain what breed the dog most likely is and why.",
    "Mention the background scenery and any objects visible around the animal.",
    "Comment on the dog's posture and what mood it seems to be in.",
    "Point out whether the dog is wearing a collar, a leash, or any accessory.",
    "Describe the lighting, the time of day, and the overall atmosphere of the scene.",
    "Speculate about where this photograph might have been taken and by whom.",
    "Finally, write a short friendly caption suitable for a social media post.",
]


def _make_prompt_of_length(base_word: str, n_words: int) -> str:
    """Build a coherent dog-related prompt of approximately `n_words` words.

    We keep adding sentences from the pool until we reach the target length,
    then trim to exactly `n_words`. This gives realistic text (no "dog dog dog")
    while still letting us sweep prompt length for the scaling study.
    """
    n_words = max(1, n_words)
    words: List[str] = []
    i = 0
    while len(words) < n_words:
        sentence = _DOG_SENTENCES[i % len(_DOG_SENTENCES)]
        words.extend(sentence.split())
        i += 1
    return " ".join(words[:n_words])


def main(
    model_path: str = None,
    image_file_path: str = None,
    prompt: str = "this building is",
    max_tokens_to_generate: int = 100,
    sweep_prompt_lengths: str = None,   # e.g. "1,8,32,64,128" (word counts)
    warmup: bool = True,
    do_sample: bool = False,
    temperature: float = 0.8,
    top_p: float = 0.9,
    ignore_eos: bool = False,           # force full-length generation for clean ITL samples
    only_cpu: bool = False,
    json_out: str = None,
):
    device = "cpu"
    if not only_cpu:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
    print("Device in use:", device)

    # -------- Model load time -------- #
    print("Loading model ...")
    _sync(device)
    load_start = _now()
    model, tokenizer = load_hf_model(model_path, device)
    model = model.to(device).eval()
    _sync(device)
    model_load_latency = _now() - load_start
    print(f"Model load latency: {model_load_latency:.2f} s")

    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    image = Image.open(image_file_path).convert("RGB")

    # -------- Optional warmup (excludes lazy init / cudnn autotune from timings) -------- #
    if warmup:
        print("Warmup run (not measured) ...")
        with torch.no_grad():
            run_generation(
                model, processor, device, prompt, image,
                max_tokens_to_generate=4,
                do_sample=do_sample, temperature=temperature, top_p=top_p,
            )
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        gc.collect()

    results: List[RunResult] = []

    prompts: List[str]
    if sweep_prompt_lengths:
        # `fire` may hand us a tuple/list (e.g. "1,8,32") or an int/str.
        if isinstance(sweep_prompt_lengths, (list, tuple)):
            lengths = [int(x) for x in sweep_prompt_lengths]
        else:
            lengths = [int(x) for x in str(sweep_prompt_lengths).split(",") if str(x).strip()]
        prompts = [_make_prompt_of_length("dog", n) for n in lengths]
    else:
        prompts = [prompt]

    with torch.no_grad():
        for p in prompts:
            res = run_generation(
                model, processor, device, p, image,
                max_tokens_to_generate=max_tokens_to_generate,
                do_sample=do_sample, temperature=temperature, top_p=top_p,
                ignore_eos=ignore_eos,
            )
            results.append(res)
            print_result(res)

    if len(results) > 1:
        print_sweep_table(results)

    if device == "cuda":
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"\nPeak CUDA memory allocated: {peak:.2f} MB")

    if json_out:
        payload = {
            "device": device,
            "model_load_latency_s": model_load_latency,
            "runs": [r.summary_dict() | {"output": r.generated_text} for r in results],
        }
        with open(json_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote metrics to {json_out}")


if __name__ == "__main__":
    fire.Fire(main)
