"""
Merge results/*.json into one comparison table, relative to the unquantized run.

    python summarize.py                 # prints a Markdown table
    python summarize.py --baseline bf16 --out results/summary.md

Without --baseline, the run with no quantization is the baseline.
"""

import argparse
import glob
import json
import os


def load(results_dir):
    runs = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            r = json.load(f)
        runs[r["label"]] = r
    return runs


def table(runs, baseline):
    base = runs.get(baseline)
    batch_sizes = sorted({t["batch_size"] for r in runs.values() for t in r["throughput"]})
    head = (["Run", "Quantization"] + [f"Tok/s @ batch {b}" for b in batch_sizes]
            + ["TTFT (ms)", "ms / token", "Perplexity"])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]

    order = sorted(runs, key=lambda name: (name != baseline, name))   # baseline row first
    for label in order:
        r = runs[label]
        tput = {t["batch_size"]: t["output_tokens_per_s"] for t in r["throughput"]}
        base_tput = {t["batch_size"]: t["output_tokens_per_s"] for t in base["throughput"]} if base else {}
        cells = [label, str(r.get("quantization") or "none")]
        for b in batch_sizes:
            if b not in tput:
                cells.append("-")
            elif base and label != baseline and b in base_tput:
                cells.append(f"{tput[b]:,.0f} ({tput[b] / base_tput[b]:.2f}x)")
            else:
                cells.append(f"{tput[b]:,.0f}")
        lat = r["latency"]
        cells += [f"{lat['ttft_ms']:.1f}", f"{lat['tpot_ms']:.2f}"]
        ppl = r.get("perplexity")
        if ppl is None:
            cells.append("-")
        elif base and label != baseline and base.get("perplexity"):
            cells.append(f"{ppl:.3f} ({100 * (ppl / base['perplexity'] - 1):+.1f}%)")
        else:
            cells.append(f"{ppl:.3f}")
        lines.append("| " + " | ".join(cells) + " |")

    first = next(iter(runs.values()))
    note = (f"\nGPU: {first['gpu']}. {first['input_len']}-token prompts, "
            f"{first['output_len']} output tokens. "
            f"Ratios and perplexity changes are relative to `{baseline}`.")
    return "\n".join(lines) + note


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--baseline", help="label of the run to compare against")
    p.add_argument("--out")
    args = p.parse_args()

    runs = load(args.results_dir)
    if not runs:
        raise SystemExit(f"No results in {args.results_dir}/. Run bench_vllm.py first.")
    baseline = args.baseline or next(
        (name for name, r in runs.items() if not r.get("quantization")), None)
    if baseline not in runs:
        raise SystemExit(f"No baseline run found. Pass --baseline, one of: {', '.join(runs)}")
    md = table(runs, baseline)
    print(md)
    if args.out:
        with open(args.out, "w") as f:
            f.write(md + "\n")


if __name__ == "__main__":
    main()
