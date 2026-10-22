| Run | Quantization | Tok/s @ batch 1 | Tok/s @ batch 4 | Tok/s @ batch 16 | Tok/s @ batch 64 | TTFT (ms) | ms / token | Perplexity |
|---|---|---|---|---|---|---|---|---|
| bf16 | none | 81 | 311 | 987 | 2,148 | 37.0 | 12.04 | 8.123 |
| awq-int4 | auto_awq | 183 (2.24x) | 603 (1.94x) | 1,502 (1.52x) | 2,189 (1.02x) | 42.9 | 5.18 | 8.665 (+6.7%) |
| gptq-int8 | auto_gptq | 130 (1.60x) | 444 (1.43x) | 1,199 (1.21x) | 1,971 (0.92x) | 53.7 | 7.32 | 8.133 (+0.1%) |
GPU: NVIDIA A100-SXM4-40GB. 512-token prompts, 128 output tokens. Ratios and perplexity changes are relative to `bf16`.
