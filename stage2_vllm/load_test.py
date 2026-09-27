"""
Stage 2 — Concurrent load test for a vLLM-served PaliGemma (or any OpenAI-compatible
multimodal endpoint).

It fires a configurable number of *concurrent* requests at each concurrency level
and measures, per request:

    - TTFT  (Time To First Token)      -> from streaming: time of first chunk
    - ITL   (Inter-Token Latency)       -> mean gap between streamed chunks
    - end-to-end latency
    - generated token count (approx, from chunk count)

and, per concurrency level:

    - system throughput (total tokens / wall-clock)
    - requests/sec
    - p50 / p90 / p99 of TTFT and end-to-end latency

This lets you plot throughput-vs-concurrency and p99-latency-vs-concurrency to
find the saturation "knee" — the core of the concurrency study in BENCHMARKING.md.

Prereqs
-------
    pip install openai pillow
    # and serve the model, e.g.:
    vllm serve google/paligemma-3b-pt-224 --dtype bfloat16 --max-model-len 2048

Usage
-----
    python stage2_vllm/load_test.py \
        --base_url http://localhost:8000/v1 \
        --model google/paligemma-3b-pt-224 \
        --image test_images/pic1.jpeg \
        --prompt "describe this image" \
        --max_tokens 64 \
        --concurrency 1,2,4,8,16 \
        --requests_per_level 32
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import statistics
import time
from dataclasses import dataclass, field
from typing import List

import fire


def _encode_image(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


@dataclass
class RequestMetrics:
    ttft_s: float = 0.0
    e2e_s: float = 0.0
    chunks: int = 0            # ~ number of streamed tokens
    ok: bool = True
    error: str = ""

    @property
    def itl_ms(self) -> float:
        # inter-token latency after the first token
        if self.chunks <= 1:
            return 0.0
        return ((self.e2e_s - self.ttft_s) / (self.chunks - 1)) * 1e3


@dataclass
class LevelResult:
    concurrency: int = 0
    wall_clock_s: float = 0.0
    metrics: List[RequestMetrics] = field(default_factory=list)

    @property
    def ok_metrics(self) -> List[RequestMetrics]:
        return [m for m in self.metrics if m.ok]

    @property
    def total_chunks(self) -> int:
        return sum(m.chunks for m in self.ok_metrics)

    def summary(self) -> dict:
        ok = self.ok_metrics
        ttfts = [m.ttft_s for m in ok]
        e2es = [m.e2e_s for m in ok]
        itls = [m.itl_ms for m in ok]
        return {
            "concurrency": self.concurrency,
            "requests": len(self.metrics),
            "failed": len(self.metrics) - len(ok),
            "wall_clock_s": round(self.wall_clock_s, 3),
            "req_per_s": round(len(ok) / self.wall_clock_s, 3) if self.wall_clock_s else 0.0,
            "system_tokens_per_s": round(self.total_chunks / self.wall_clock_s, 2) if self.wall_clock_s else 0.0,
            "ttft_p50_s": round(_percentile(ttfts, 50), 4),
            "ttft_p99_s": round(_percentile(ttfts, 99), 4),
            "e2e_p50_s": round(_percentile(e2es, 50), 4),
            "e2e_p99_s": round(_percentile(e2es, 99), 4),
            "itl_mean_ms": round(statistics.mean(itls), 2) if itls else 0.0,
        }


async def _one_request(client, model, prompt, image_data_url, max_tokens, temperature) -> RequestMetrics:
    from openai import AsyncOpenAI  # noqa: F401  (client is already AsyncOpenAI)

    m = RequestMetrics()
    start = time.perf_counter()
    first_token_time = None
    chunks = 0
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                chunks += 1
        end = time.perf_counter()
        m.ttft_s = (first_token_time - start) if first_token_time else (end - start)
        m.e2e_s = end - start
        m.chunks = chunks
        m.ok = True
    except Exception as e:  # noqa: BLE001
        m.ok = False
        m.error = str(e)
        m.e2e_s = time.perf_counter() - start
    return m


async def _run_level(client, model, prompt, image_data_url, max_tokens,
                     temperature, concurrency, requests_per_level) -> LevelResult:
    result = LevelResult(concurrency=concurrency)
    sem = asyncio.Semaphore(concurrency)

    async def _guarded():
        async with sem:
            return await _one_request(client, model, prompt, image_data_url, max_tokens, temperature)

    wall_start = time.perf_counter()
    tasks = [asyncio.create_task(_guarded()) for _ in range(requests_per_level)]
    result.metrics = await asyncio.gather(*tasks)
    result.wall_clock_s = time.perf_counter() - wall_start
    return result


async def _amain(base_url, api_key, model, image, prompt, max_tokens,
                 temperature, concurrency_levels, requests_per_level):
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    image_data_url = _encode_image(image)

    print(f"Endpoint : {base_url}")
    print(f"Model    : {model}")
    print(f"Prompt   : {prompt!r}")
    print(f"Levels   : {concurrency_levels}")
    print(f"Reqs/lvl : {requests_per_level}\n")

    header = (
        f"{'conc':>5} {'req/s':>8} {'sys_tok/s':>10} "
        f"{'ttft_p50':>9} {'ttft_p99':>9} {'e2e_p50':>8} {'e2e_p99':>8} "
        f"{'itl_ms':>7} {'failed':>7}"
    )
    print(header)
    print("-" * len(header))

    all_summaries = []
    for c in concurrency_levels:
        level = await _run_level(
            client, model, prompt, image_data_url, max_tokens,
            temperature, c, requests_per_level,
        )
        s = level.summary()
        all_summaries.append(s)
        print(
            f"{s['concurrency']:>5} {s['req_per_s']:>8.2f} {s['system_tokens_per_s']:>10.2f} "
            f"{s['ttft_p50_s']:>9.4f} {s['ttft_p99_s']:>9.4f} "
            f"{s['e2e_p50_s']:>8.4f} {s['e2e_p99_s']:>8.4f} "
            f"{s['itl_mean_ms']:>7.2f} {s['failed']:>7}"
        )

    print("\nLook for the 'knee': the concurrency where system_tok/s stops rising")
    print("but ttft_p99 / e2e_p99 keep climbing -> that's your saturation point.")
    return all_summaries


def main(
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    model: str = "google/paligemma-3b-pt-224",
    image: str = "test_images/pic1.jpeg",
    prompt: str = "describe this image",
    max_tokens: int = 64,
    temperature: float = 0.0,
    concurrency: str = "1,2,4,8,16",
    requests_per_level: int = 32,
):
    levels = [int(x) for x in str(concurrency).split(",") if x.strip()]
    asyncio.run(
        _amain(
            base_url, api_key, model, image, prompt, max_tokens,
            temperature, levels, requests_per_level,
        )
    )


if __name__ == "__main__":
    fire.Fire(main)
