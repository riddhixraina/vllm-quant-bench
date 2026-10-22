"""
Benchmark one model on vLLM: throughput across batch sizes, single-request latency,
and perplexity. Run once per model (FP16, AWQ, GPTQ), then summarize.py compares them.

    python bench_vllm.py --model Qwen/Qwen2.5-1.5B-Instruct --label fp16
    python bench_vllm.py --model Qwen/Qwen2.5-1.5B-Instruct-AWQ --label awq-int4
    python bench_vllm.py --model Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8 --label gptq-int8

vLLM reads the quantization method from the model's config, so there's no flag for it.
On an A100 it picks the faster Marlin kernels for AWQ and GPTQ automatically.
"""

import argparse
import json
import math
import os
import random
import time


def make_prompts(tokenizer, n, input_len, seed=0):
    """n prompts of exactly input_len token IDs, drawn from ordinary vocabulary."""
    rng = random.Random(seed)
    vocab = list(range(1000, min(tokenizer.vocab_size, 30000)))
    return [{"prompt_token_ids": [rng.choice(vocab) for _ in range(input_len)]} for _ in range(n)]


def timed_generate(llm, prompts, params):
    start = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    return time.perf_counter() - start, outputs


def prompt_token_logprobs(output):
    """Log-probabilities of each actual prompt token after the first.

    Handles both return formats vLLM has used: {id: Logprob} and {id: float}.
    """
    values = []
    for token_id, position in zip(output.prompt_token_ids[1:], output.prompt_logprobs[1:]):
        lp = position[token_id]
        values.append(lp.logprob if hasattr(lp, "logprob") else float(lp))
    return values


def perplexity(llm, tokenizer, texts, chunk_len):
    """Token-level perplexity over fixed-length chunks of the evaluation text."""
    from vllm import SamplingParams

    ids = tokenizer("\n\n".join(texts))["input_ids"]
    chunks = [ids[i:i + chunk_len] for i in range(0, len(ids) - chunk_len + 1, chunk_len)]
    if not chunks:
        raise SystemExit(f"Evaluation text is shorter than one {chunk_len}-token chunk.")
    params = SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=0.0)
    outputs = llm.generate([{"prompt_token_ids": c} for c in chunks], params, use_tqdm=False)
    logprobs = [lp for out in outputs for lp in prompt_token_logprobs(out)]
    return math.exp(-sum(logprobs) / len(logprobs)), len(logprobs)


def eval_texts(args):
    if args.ppl_file:
        with open(args.ppl_file) as f:
            return [f.read()]
    from datasets import load_dataset  # pip install datasets
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return [t for t in data["text"] if t.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--label", required=True, help="short name for this run, e.g. fp16 or awq-int4")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 64])
    p.add_argument("--input-len", type=int, default=512)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--gpu-mem", type=float, default=0.85, help="share of GPU memory vLLM may use")
    p.add_argument("--dtype", default="auto", help="use 'half' on GPUs without bf16, such as a T4")
    p.add_argument("--ppl-chunk", type=int, default=1024)
    p.add_argument("--ppl-file", help="text file for perplexity instead of WikiText-2")
    p.add_argument("--skip-ppl", action="store_true")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    import torch
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, dtype=args.dtype, gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_model_len, seed=0)
    tokenizer = llm.get_tokenizer()

    try:
        quant = llm.llm_engine.model_config.quantization
    except AttributeError:
        quant = None

    # Warm-up: the first calls pay for CUDA graph capture and kernel compilation.
    warm = SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True)
    timed_generate(llm, make_prompts(tokenizer, 4, args.input_len, seed=99), warm)

    # Throughput: every request generates exactly output_len tokens (ignore_eos).
    gen = SamplingParams(max_tokens=args.output_len, temperature=0.0, ignore_eos=True)
    throughput = []
    for b in args.batch_sizes:
        seconds, outs = timed_generate(llm, make_prompts(tokenizer, b, args.input_len), gen)
        out_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
        throughput.append({
            "batch_size": b,
            "seconds": seconds,
            "output_tokens_per_s": out_tokens / seconds,
            "total_tokens_per_s": (out_tokens + b * args.input_len) / seconds,
        })
        print(f"batch {b:>4}: {out_tokens / seconds:8.1f} output tok/s")

    # Latency for one request. Time to first token is measured as a 1-token generation;
    # the time per output token is the rest of a full generation spread over its tokens.
    one = make_prompts(tokenizer, 1, args.input_len, seed=7)
    ttft = min(timed_generate(llm, one, SamplingParams(max_tokens=1, temperature=0.0))[0]
               for _ in range(3))
    e2e = min(timed_generate(llm, one, gen)[0] for _ in range(3))
    tpot = (e2e - ttft) / max(args.output_len - 1, 1)
    print(f"latency: TTFT {ttft * 1e3:.1f} ms, {tpot * 1e3:.2f} ms per output token")

    ppl = n_tokens = None
    if not args.skip_ppl:
        ppl, n_tokens = perplexity(llm, tokenizer, eval_texts(args), args.ppl_chunk)
        print(f"perplexity: {ppl:.3f} over {n_tokens:,} tokens")

    result = {
        "label": args.label, "model": args.model, "quantization": quant,
        "gpu": torch.cuda.get_device_name(), "dtype": args.dtype,
        "input_len": args.input_len, "output_len": args.output_len,
        "throughput": throughput,
        "latency": {"ttft_ms": ttft * 1e3, "tpot_ms": tpot * 1e3, "e2e_ms": e2e * 1e3},
        "perplexity": ppl, "perplexity_tokens": n_tokens,
        "gpu_memory_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{args.label}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
