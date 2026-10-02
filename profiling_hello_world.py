"""
Hello-World of PyTorch profiling.

The ENTIRE profiler API is ~5 lines:

    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        <run some code>
    print(prof.key_averages().table(sort_by="self_cuda_time_total"))

That's it. Everything else is just picking what to measure and how to sort.

This file shows three tiny experiments so you can SEE the ideas from
PERF_ENGINEERING.md with your own eyes:

  1. A single big matrix-matrix multiply  -> GEMM kernel (compute-bound)
  2. A matrix-VECTOR multiply             -> GEMV kernel (memory-bound)  <-- the "flip"
  3. Many tiny ops                        -> lots of kernel launches (launch overhead)

Run:
    python profiling_hello_world.py
"""

import os
import torch
from torch.profiler import profile, ProfilerActivity, record_function

OUT_DIR = "profiling_hello_world_out"


def banner(msg):
    print("\n" + "=" * 70 + f"\n{msg}\n" + "=" * 70)


def _top_kernel_name(prof):
    """Return the name of the real GPU kernel with the most GPU time.

    We skip Python/aten wrappers and our own record_function labels so we report
    the actual device kernel (e.g. 'ampere_sgemm...' or 'gemv...'), not 'aten::mm'.
    """
    skip_prefixes = ("aten::", "cuda", "void at::native::record")
    skip_exact = {"matmul_MxM", "matmul_Mxv", "fifty_tiny_ops"}
    best_name, best_time = "(none)", 0
    for evt in prof.key_averages():
        dev_time = (getattr(evt, "self_device_time_total", 0)
                    or getattr(evt, "cuda_time_total", 0))
        if not dev_time:
            continue
        name = evt.key
        if name in skip_exact or name.startswith(skip_prefixes):
            continue
        if dev_time > best_time:
            best_time, best_name = dev_time, name
    return best_name, best_time


def _count_launches(prof):
    count = 0
    for evt in prof.key_averages():
        dev_time = (getattr(evt, "self_device_time_total", 0)
                    or getattr(evt, "cuda_time_total", 0))
        if dev_time and evt.count:
            count += evt.count
    return count


def _save_experiment(name, title, takeaway, prof):
    """Write one experiment's full table + a plain-English summary to its own file."""
    os.makedirs(OUT_DIR, exist_ok=True)
    table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=12)
    top_name, top_us = _top_kernel_name(prof)
    launches = _count_launches(prof)

    txt_path = os.path.join(OUT_DIR, f"{name}.txt")
    trace_path = os.path.join(OUT_DIR, f"{name}.json")
    prof.export_chrome_trace(trace_path)

    with open(txt_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(title + "\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"HOTSPOT KERNEL : {top_name}\n")
        f.write(f"  (spent {top_us/1000:.3f} ms on the GPU)\n")
        f.write(f"TOTAL GPU KERNEL LAUNCHES : {launches}\n\n")
        f.write("TAKEAWAY:\n")
        f.write("  " + takeaway.replace("\n", "\n  ") + "\n\n")
        f.write("FULL KERNEL TABLE (sorted by GPU self-time):\n")
        f.write(table + "\n")
        f.write(f"\nChrome trace: {trace_path}  (open in https://ui.perfetto.dev)\n")

    # Concise console summary so the terminal stays readable.
    print(f"  hotspot kernel : {top_name}")
    print(f"  GPU launches   : {launches}")
    print(f"  details written: {txt_path}")
    print(f"  trace written  : {trace_path}")


def main():
    if not torch.cuda.is_available():
        print("No CUDA GPU found — this demo needs a GPU to show kernels.")
        return

    dev = "cuda"
    print("GPU:", torch.cuda.get_device_name(0))

    # ----------------------------------------------------------------- #
    # Make some data. A 'matrix' has many rows; a 'vector' has one row.
    # ----------------------------------------------------------------- #
    W = torch.randn(4096, 4096, device=dev)      # a weight matrix
    X_matrix = torch.randn(4096, 4096, device=dev)  # many tokens  -> MxM
    x_vector = torch.randn(1, 4096, device=dev)     # one token    -> Mxv

    # Warmup (first CUDA call triggers lazy init / autotuning — never time it).
    for _ in range(3):
        _ = X_matrix @ W
        _ = x_vector @ W
    torch.cuda.synchronize()

    # ================================================================= #
    # EXPERIMENT 1 — matrix x matrix  => GEMM (compute-bound)
    # ================================================================= #
    banner("1) MATRIX x MATRIX  -> expect an *sgemm* (GEMM) kernel")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(20):
            with record_function("matmul_MxM"):   # a custom label you choose
                y = X_matrix @ W
        torch.cuda.synchronize()
    _save_experiment(
        "exp1_matrix_x_matrix",
        "EXPERIMENT 1 — MATRIX x MATRIX (the PREFILL regime)",
        "Input is a MATRIX (many rows), so cuBLAS/cutlass picks a GEMM kernel\n"
        "(name contains 'sgemm'). GEMM reuses each weight across many rows =\n"
        "high arithmetic intensity = COMPUTE-bound. This is what prefill looks like.",
        prof,
    )

    # ================================================================= #
    # EXPERIMENT 2 — matrix x vector => GEMV (memory-bound)  <-- the flip
    # ================================================================= #
    banner("2) MATRIX x VECTOR  -> expect a *gemv* kernel (the decode regime)")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(20):
            with record_function("matmul_Mxv"):
                y = x_vector @ W
        torch.cuda.synchronize()
    _save_experiment(
        "exp2_matrix_x_vector",
        "EXPERIMENT 2 — MATRIX x VECTOR (the DECODE regime)  <-- the 'flip'",
        "Same weight matrix, but input is now a VECTOR (one row). cuBLAS picks a\n"
        "GEMV kernel (name contains 'gemv'). GEMV reads all the weights to do one\n"
        "row of math = low arithmetic intensity = MEMORY-bound. This is decode.\n"
        "Seeing 'gemv' instead of 'sgemm' is literally how you spot the decode regime.",
        prof,
    )

    # ================================================================= #
    # EXPERIMENT 3 — many tiny ops => many kernel launches (overhead)
    # ================================================================= #
    banner("3) MANY TINY OPS  -> many kernel launches (launch overhead)")
    small = torch.randn(1024, device=dev)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function("fifty_tiny_ops"):
            for _ in range(50):
                # Each line launches its own kernel (unfused):
                small = small * 1.001
                small = small + 0.5
                small = torch.relu(small)
        torch.cuda.synchronize()
    _save_experiment(
        "exp3_many_tiny_ops",
        "EXPERIMENT 3 — MANY TINY OPS (launch overhead)",
        "150 small elementwise ops (50 mul + 50 add + 50 relu) each launch their\n"
        "OWN kernel -> ~150 kernels. The GPU finishes each instantly then waits for\n"
        "the CPU to launch the next = LAUNCH-bound. This is why our real decode step\n"
        "fires thousands of kernels. torch.compile would FUSE these into a handful.",
        prof,
    )

    # ================================================================= #
    # EXPERIMENT 4 — SAME tiny ops, but with torch.compile => FUSION
    # ================================================================= #
    banner("4) torch.compile  -> fuses the tiny ops into few kernels")

    # --- How to write torch.compile (this is the whole API) ---
    # 1. Put the work in a function.
    def tiny_ops(x):
        for _ in range(50):
            x = x * 1.001
            x = x + 0.5
            x = torch.relu(x)
        return x

    # 2. Wrap it ONCE with torch.compile. It returns a new callable that,
    #    on first call, traces the function, fuses the ops, and generates
    #    optimized (Triton) kernels. Call it exactly like the original.
    compiled_tiny_ops = torch.compile(tiny_ops)

    # 3. WARM UP the compiled function (the FIRST call does the slow
    #    compilation — never profile that). Subsequent calls are fast.
    small = torch.randn(1024, device=dev)
    for _ in range(3):
        _ = compiled_tiny_ops(small)
    torch.cuda.synchronize()

    # 4. Now profile the compiled version — same code as exp3, just compiled.
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function("fifty_tiny_ops_compiled"):
            y = compiled_tiny_ops(small)
        torch.cuda.synchronize()
    _save_experiment(
        "exp4_torch_compile",
        "EXPERIMENT 4 — SAME tiny ops, but torch.compile (FUSION)",
        "Identical math to exp3, but torch.compile FUSES the 150 elementwise ops\n"
        "into just a few Triton kernels. Compare TOTAL GPU KERNEL LAUNCHES here vs\n"
        "exp3 (311) — it should drop dramatically. Fewer launches = less CPU\n"
        "dispatch overhead = the fix for launch-bound decode. This is exactly what\n"
        "torch.compile would do to the real model's thousands of per-token kernels.",
        prof,
    )

    # ----------------------------------------------------------------- #
    # Final: a clean index so you know which file is which.
    # ----------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("DONE. Open these to study each experiment one at a time:")
    print(f"  {OUT_DIR}/exp1_matrix_x_matrix.txt   (GEMM / prefill regime)")
    print(f"  {OUT_DIR}/exp2_matrix_x_vector.txt   (GEMV / decode regime)")
    print(f"  {OUT_DIR}/exp3_many_tiny_ops.txt     (launch overhead)")
    print(f"  {OUT_DIR}/exp4_torch_compile.txt     (torch.compile fuses them)")
    print(f"  matching .json traces -> open in https://ui.perfetto.dev")
    print("=" * 70)


if __name__ == "__main__":
    main()
