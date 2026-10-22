| Run | Quantization | Tok/s @ batch 1 | Tok/s @ batch 4 | Tok/s @ batch 16 | Tok/s @ batch 64 | TTFT (ms) | ms / token | Perplexity |
|---|---|---|---|---|---|---|---|---|
| bf16 | none | 77 | 251 | 566 | 800 | 126.4 | 12.09 | - |
| awq-int4 | auto_awq | 156 (2.02x) | 381 (1.52x) | 613 (1.08x) | 688 (0.86x) | 160.4 | 5.22 | - |
| gptq-int8 | auto_gptq | 114 (1.48x) | 301 (1.20x) | 524 (0.93x) | 620 (0.77x) | 183.5 | 7.38 | - |
GPU: NVIDIA A100-SXM4-40GB. 2048-token prompts, 128 output tokens. Ratios and perplexity changes are relative to `bf16`.
