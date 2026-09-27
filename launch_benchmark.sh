#!/bin/bash
# Benchmark PaliGemma inference: model-load, prefill, decode, and KV-cache metrics.
# Sweeps several prompt lengths so you can see how prefill cost and KV-cache
# memory grow with the prompt.

MODEL_PATH="$HOME/projects/paligemma-weights/paligemma-3b-pt-224"
IMAGE_FILE_PATH="test_images/pic1.jpeg"
PROMPT="this building is"
MAX_TOKENS_TO_GENERATE=100
SWEEP_PROMPT_LENGTHS="1,8,32,64,128"   # word counts; comment out for a single prompt
ONLY_CPU="False"
JSON_OUT="benchmark_results.json"

python benchmark_inference.py \
    --model_path "$MODEL_PATH" \
    --image_file_path "$IMAGE_FILE_PATH" \
    --prompt "$PROMPT" \
    --max_tokens_to_generate $MAX_TOKENS_TO_GENERATE \
    --sweep_prompt_lengths "$SWEEP_PROMPT_LENGTHS" \
    --only_cpu $ONLY_CPU \
    --json_out "$JSON_OUT"
