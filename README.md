# LLM Inference Quantization Benchmark on vLLM

How much faster does a quantized LLM serve, and what does it cost in quality? This benchmarks Qwen2.5 on vLLM in three forms:

| Label | Weights | Model |
|---|---|---|
| `bf16` | 16-bit (baseline) | `Qwen/Qwen2.5-<size>-Instruct` |
| `awq-int4` | 4-bit AWQ | `Qwen/Qwen2.5-<size>-Instruct-AWQ` |
| `gptq-int8` | 8-bit GPTQ | `Qwen/Qwen2.5-<size>-Instruct-GPTQ-Int8` |

The baseline runs in bfloat16, the model's native type. On a T4, which has no bfloat16, it runs in float16 and is labeled `fp16` instead.

For each one it measures:
- **Throughput** in output tokens per second at batch sizes 1, 4, 16 and 64, with 512-token prompts and 128 generated tokens.
- **Latency** for a single request: time to first token, and time per output token after that.
- **Quality** as perplexity on WikiText-2. Lower is better, and the interesting number is how much it rises versus the 16-bit baseline.

## Results

Qwen2.5-7B on one A100, 512-token prompts, 128 output tokens:

- **At batch 1, AWQ 4-bit runs 2.2× faster than BF16.** Decoding one request is memory-bound, and 4-bit weights are a quarter the size.
- **At batch 64 the gain shrinks to about 1.07×**, and 8-bit GPTQ ends up about 4% slower than BF16. Large batches are compute-bound, so unpacking quantized weights becomes overhead.
- **AWQ raises perplexity by 6.7%.** 8-bit GPTQ is near-lossless.

## Run it

You need an NVIDIA GPU. The 1.5B model fits a free Colab T4 (16 GB). The 7B model needs about 24 GB or more, such as an A100.

```bash
pip install -r requirements.txt
./run_all.sh 1.5B        # or 7B on an A100
```

This runs each model in its own process (vLLM doesn't fully free GPU memory between models in one process), saves `results/<label>.json`, and writes `results/summary.md`:

```
| Run       | Quantization | Tok/s @ batch 1 | ... | TTFT (ms) | ms / token | Perplexity    |
| bf16      | none         | ...             |     |           |            | ...           |
| awq-int4  | awq_marlin   | ... (2.20x)     |     |           |            | ... (+6.7%)   |
```

To run a single model: `python bench_vllm.py --model Qwen/Qwen2.5-1.5B-Instruct-AWQ --label awq-int4`. On a T4, add `--dtype half`. `run_all.sh` handles this for you.

Anything after the model size is passed to every run. To test long prompts and keep those results apart from the first run:

```bash
OUT=results-2048 ./run_all.sh 7B --input-len 2048 --skip-ppl
```

## What to expect, and why

- **Batch 1 is where 4-bit should win most.** Generating one token at a time is memory-bound: each step reads every weight once to do very little math. 4-bit weights are a quarter the size of 16-bit ones, so each step moves a quarter of the bytes.
- **The gain should shrink at batch 64.** With many requests per step, each weight read is reused across the batch, so the GPU becomes compute-bound, and unpacking 4-bit weights back to 16-bit becomes overhead.
- **Time to first token should improve much less than decoding.** Prefill runs the whole prompt through the model in one pass, so it's compute-bound, and weight-only quantization doesn't reduce the math. The longer the prompt, the clearer this gets. With 2048-token prompts, 4-bit can even be slower than 16-bit here.
- **Smaller weights leave more memory for the KV cache**, so the server can hold more requests at once. That shows up in larger batches.
- **Kernels matter.** On an A100, vLLM picks the Marlin kernels for AWQ and GPTQ automatically; the `Quantization` column shows which one ran. On a T4 it falls back to slower kernels, so the gap between 16-bit and 4-bit will look different there.

## Measurement notes

- **Prefix caching is off.** vLLM turns it on by default. With it on, a prompt the engine has already seen skips most of its prefill. Timing the same prompt three times then measures a cache hit, which on an A100 shows up as a time to first token below what a real 512-token prefill of a 7B model can physically take (about 6.7 TFLOP, so at least 21 ms at the A100's 312 TFLOPS peak).
- **Every batch size gets its own prompts**, so no throughput run reuses another run's prompts.
- **Prompts are random token IDs** of exactly the requested length, and every request generates exactly 128 tokens (`ignore_eos`), so all three models do the same work.
