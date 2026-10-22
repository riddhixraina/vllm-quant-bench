#!/usr/bin/env bash
# Run FP16, AWQ 4-bit and GPTQ 8-bit versions of one Qwen2.5 model, then compare.
#
#   ./run_all.sh 1.5B          # fits a 16 GB GPU such as a free Colab T4
#   ./run_all.sh 7B            # needs about 24 GB or more, for example an A100
#
# Each model runs in its own process so vLLM frees GPU memory between runs.
set -euo pipefail

SIZE="${1:-1.5B}"
BASE="Qwen/Qwen2.5-${SIZE}-Instruct"

# T4s and other pre-Ampere GPUs have no bfloat16, so fall back to fp16 there.
CAP=$(python -c "import torch; print(torch.cuda.get_device_capability()[0])")
DTYPE=auto
if [ "$CAP" -lt 8 ]; then DTYPE=half; fi

python bench_vllm.py --model "$BASE"                 --label fp16      --dtype "$DTYPE"
python bench_vllm.py --model "${BASE}-AWQ"           --label awq-int4  --dtype "$DTYPE"
python bench_vllm.py --model "${BASE}-GPTQ-Int8"     --label gptq-int8 --dtype "$DTYPE"

python summarize.py --baseline fp16 --out results/summary.md
