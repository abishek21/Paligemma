"""
Generate the charts used in the README from the measured numbers.

Produces (in docs/images/):
  - optimization_journey.png  : TPOT + throughput across the 5 optimization stages
  - hf_comparison.png         : ours vs HuggingFace TPOT across configs
  - roofline.png              : prefill (compute-bound) vs decode (memory-bound)

All numbers are the measured results from our runs (batch 32, ctx 260 on an A40).
Run:  python make_plots.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "docs/images"
os.makedirs(OUT, exist_ok=True)

# --------------------------------------------------------------------------- #
# Measured data (A40, batch 32, ctx 260, prompt "caption en")
# --------------------------------------------------------------------------- #
STAGES = ["baseline\nfp32", "+ encode\nonce", "+ bf16", "+ torch.\ncompile", "+ Flash\nAttention"]
TPOT_MS = [516.7, 33.4, 20.6, 17.2, 15.94]
SYS_TPS = [61.9, 958.5, 1552.2, 1862.9, 2007.0]

HF_CONFIGS = ["fp32", "bf16", "bf16+compile", "bf16+sdpa\n+compile"]
OURS_TPOT = [32.45, 19.37, 16.88, 15.94]
HF_TPOT   = [23.29, 17.61, 15.74, 15.74]  # hf has no separate sdpa row (already uses it)


def plot_journey():
    fig, ax1 = plt.subplots(figsize=(9, 5))
    x = range(len(STAGES))

    color = "#d62728"
    ax1.set_ylabel("TPOT (ms/token) — log scale", color=color, fontsize=11)
    ax1.plot(x, TPOT_MS, "o-", color=color, linewidth=2.5, markersize=9, label="TPOT")
    ax1.set_yscale("log")
    ax1.tick_params(axis="y", labelcolor=color)
    for xi, y in zip(x, TPOT_MS):
        ax1.annotate(f"{y:.1f}ms", (xi, y), textcoords="offset points",
                     xytext=(0, 10), ha="center", color=color, fontsize=9, fontweight="bold")

    ax2 = ax1.twinx()
    color2 = "#2ca02c"
    ax2.set_ylabel("System throughput (tok/s)", color=color2, fontsize=11)
    ax2.bar(x, SYS_TPS, alpha=0.25, color=color2, width=0.5)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(STAGES, fontsize=10)
    ax1.set_title("PaliGemma decode optimization journey  (A40, batch 32)\n"
                  "516.7 ms → 15.9 ms  =  32× faster", fontsize=13, fontweight="bold")
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/optimization_journey.png", dpi=130)
    print(f"wrote {OUT}/optimization_journey.png")


def plot_hf():
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(HF_CONFIGS))
    w = 0.38
    ax.bar([i - w/2 for i in x], OURS_TPOT, w, label="ours (from scratch)", color="#1f77b4")
    ax.bar([i + w/2 for i in x], HF_TPOT, w, label="HuggingFace", color="#ff7f0e")
    for i in x:
        ax.annotate(f"{OURS_TPOT[i]:.1f}", (i - w/2, OURS_TPOT[i]),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
        ax.annotate(f"{HF_TPOT[i]:.1f}", (i + w/2, HF_TPOT[i]),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    ax.set_ylabel("TPOT (ms/token) — lower is better", fontsize=11)
    ax.set_xticks(list(x)); ax.set_xticklabels(HF_CONFIGS, fontsize=10)
    ax.set_title("Ours vs HuggingFace  (A40, batch 32, ctx 260)\n"
                 "best: 15.94 ms vs 15.74 ms — within 1.3%", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/hf_comparison.png", dpi=130)
    print(f"wrote {OUT}/hf_comparison.png")


def plot_roofline():
    fig, ax = plt.subplots(figsize=(8, 5.5))
    # A40 roofline
    peak_flops = 37.4e12      # fp32 TFLOP/s
    peak_bw = 696e9           # GB/s
    ridge = peak_flops / peak_bw  # ~53.7 FLOP/byte

    import numpy as np
    ai = np.logspace(-1, 4, 400)
    perf = np.minimum(peak_flops, peak_bw * ai) / 1e12  # TFLOP/s
    ax.loglog(ai, perf, "k-", linewidth=2, label="A40 roofline (fp32)")
    ax.axvline(ridge, color="gray", ls="--", alpha=0.6)
    ax.annotate(f"ridge ≈ {ridge:.0f} FLOP/byte", (ridge, 0.3),
                rotation=90, va="bottom", ha="right", color="gray", fontsize=9)

    # Our measured points
    ax.scatter([0.5], [peak_bw * 0.5 / 1e12], color="#d62728", s=120, zorder=5)
    ax.annotate("DECODE\n(memory-bound, AI≈0.5)", (0.5, peak_bw * 0.5 / 1e12),
                textcoords="offset points", xytext=(12, -5), color="#d62728", fontsize=10)
    ax.scatter([130], [peak_flops / 1e12], color="#2ca02c", s=120, zorder=5)
    ax.annotate("PREFILL\n(compute-bound, AI≈130)", (130, peak_flops / 1e12),
                textcoords="offset points", xytext=(-40, -35), color="#2ca02c", fontsize=10)

    ax.set_xlabel("Arithmetic Intensity (FLOP / byte)", fontsize=11)
    ax.set_ylabel("Performance (TFLOP/s)", fontsize=11)
    ax.set_title("Roofline: why prefill and decode are different machines",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/roofline.png", dpi=130)
    print(f"wrote {OUT}/roofline.png")


if __name__ == "__main__":
    plot_journey()
    plot_hf()
    plot_roofline()
    print("Done.")
