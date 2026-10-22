#!/usr/bin/env bash
# Run 16-bit, AWQ 4-bit and GPTQ 8-bit versions of one Qwen2.5 model, then compare.
#
#   ./run_all.sh 1.5B          # fits a 16 GB GPU such as a free Colab T4
#   ./run_all.sh 7B            # needs about 24 GB or more, for example an A100
#
# Anything after the size goes to every bench_vllm.py run. Set OUT to keep a
# second experiment apart from the first, for example long prompts:
#
#   OUT=results-2048 ./run_all.sh 7B --input-len 2048 --skip-ppl
#
# Each model runs in its own process so vLLM frees GPU memory between runs.
set -euo pipefail

SIZE="${1:-1.5B}"
shift || true
BASE="Qwen/Qwen2.5-${SIZE}-Instruct"
OUT="${OUT:-results}"

# The baseline runs in bfloat16, the model's native type. T4s and other pre-Ampere
# GPUs have no bfloat16, so it runs in float16 there, and the label says which.
CAP=$(python -c "import torch; print(torch.cuda.get_device_capability()[0])")
if [ "$CAP" -ge 8 ]; then DTYPE=auto; BASELINE=bf16; else DTYPE=half; BASELINE=fp16; fi

python bench_vllm.py --model "$BASE"             --label "$BASELINE" --dtype "$DTYPE" --out-dir "$OUT" "$@"
python bench_vllm.py --model "${BASE}-AWQ"       --label awq-int4    --dtype "$DTYPE" --out-dir "$OUT" "$@"
python bench_vllm.py --model "${BASE}-GPTQ-Int8" --label gptq-int8   --dtype "$DTYPE" --out-dir "$OUT" "$@"

python summarize.py --results-dir "$OUT" --baseline "$BASELINE" --out "$OUT/summary.md"
