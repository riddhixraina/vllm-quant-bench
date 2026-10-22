# LLM Inference Quantization Benchmark on vLLM

How much faster does a quantized LLM serve, and what does it cost in quality? This benchmarks Qwen2.5 on vLLM in three forms:

| Label | Weights | Model |
|---|---|---|
| `fp16` | 16-bit (baseline) | `Qwen/Qwen2.5-<size>-Instruct` |
| `awq-int4` | 4-bit AWQ | `Qwen/Qwen2.5-<size>-Instruct-AWQ` |
| `gptq-int8` | 8-bit GPTQ | `Qwen/Qwen2.5-<size>-Instruct-GPTQ-Int8` |

For each one it measures:
- **Throughput** in output tokens per second at batch sizes 1, 4, 16 and 64, with 512-token prompts and 128 generated tokens.
- **Latency** for a single request: time to first token, and time per output token after that.
- **Quality** as perplexity on WikiText-2. Lower is better, and the interesting number is how much it rises versus FP16.

## Run it

You need an NVIDIA GPU. The 1.5B model fits a free Colab T4 (16 GB). The 7B model needs about 24 GB or more, such as an NYU HPC A100.

```bash
pip install -r requirements.txt
./run_all.sh 1.5B        # or 7B on an A100
```

This runs each model in its own process (vLLM doesn't fully free GPU memory between models in one process), saves `results/<label>.json`, and writes `results/summary.md`:

```
| Run       | Quantization | Tok/s @ batch 1 | ... | TTFT (ms) | ms / token | Perplexity    |
| fp16      | none         | ...             |     |           |            | ...           |
| awq-int4  | awq_marlin   | ... (1.80x)     |     |           |            | ... (+4.0%)   |
```

To run a single model: `python bench_vllm.py --model Qwen/Qwen2.5-1.5B-Instruct-AWQ --label awq-int4`. On a T4, add `--dtype half`, because T4s have no bfloat16. `run_all.sh` handles this for you.

## What to expect, and why

- **Batch 1 is where 4-bit should win most.** Generating one token at a time is memory-bound: each step reads every weight once to do very little math. 4-bit weights are a quarter the size of FP16, so each step moves a quarter of the bytes.
- **The gain should shrink at batch 64.** With many requests per step, each weight read is reused across the batch, so the GPU becomes compute-bound, and unpacking 4-bit weights back to 16-bit becomes overhead.
- **Smaller weights leave more memory for the KV cache**, so the server can hold more requests at once. That shows up in larger batches.
- **Kernels matter.** On an A100, vLLM picks the Marlin kernels for AWQ and GPTQ automatically; the `Quantization` column shows which one ran. On a T4 it falls back to slower kernels, so the gap between FP16 and 4-bit will look different there. That difference is worth explaining if you're asked.

## What to put on your resume

Fill in the brackets from `results/summary.md`, and name the GPU:

> Benchmarked Qwen2.5-[7B] on vLLM in FP16, AWQ 4-bit and GPTQ 8-bit, finding AWQ delivers [X]× FP16 throughput at batch 1 and [Y]× at batch 64 for a [Z]% perplexity increase on an [A100].

## Questions to be ready for

- **AWQ vs GPTQ?** Both quantize weights only. GPTQ picks rounding that minimizes each layer's output error using second-order information. AWQ finds the small share of weights that matter most, by looking at activation sizes, and scales them to protect them before rounding.
- **Why not quantize activations too?** That's W8A8 or FP8. It speeds up compute-bound prefill, but needs hardware support: FP8 tensor cores arrived with Hopper and Ada GPUs, not on the A100.
- **Why does time to first token barely change?** Prefill processes the whole prompt at once, so it's compute-bound, and weight-only quantization mostly speeds up the memory-bound decode steps.
- **What would you test next?** Longer contexts, where the KV cache dominates memory, KV-cache quantization (`kv_cache_dtype="fp8"`), and an accuracy benchmark beyond perplexity.
