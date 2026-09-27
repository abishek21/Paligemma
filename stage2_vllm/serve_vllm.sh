#!/bin/bash
# Stage 2: start a vLLM OpenAI-compatible server for PaliGemma.
# Requires: pip install vllm  (and a CUDA GPU).

MODEL="google/paligemma-3b-pt-224"

vllm serve "$MODEL" \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --max-num-seqs 64 \
    --gpu-memory-utilization 0.90 \
    --port 8000
# OpenAI-compatible endpoint will be at http://localhost:8000/v1
