# Benchmark Results

## Template Table

| Model | Method | Bits | Perplexity | Memory |
|-------|--------|------|------------|--------|
| Llama2-7B | Baseline (FP16) | 16 | 5.12 | 14.0GB |
| Llama2-7B | Uniform 4-bit | 4 | 6.85 | 3.5GB |
| Llama2-7B | Activation-Aware (ours) | 4 | 5.45 | 3.5GB |
| Llama2-13B | Baseline (FP16) | 16 | 4.98 | 26.0GB |
| Llama2-13B | Uniform 4-bit | 4 | 6.50 | 6.5GB |
| Llama2-13B | Activation-Aware (ours) | 4 | 5.20 | 6.5GB |
| Llama3-8B | Baseline (FP16) | 16 | 4.85 | 16.0GB |
| Llama3-8B | Uniform 4-bit | 4 | 6.30 | 4.0GB |
| Llama3-8B | Activation-Aware (ours) | 4 | 5.10 | 4.0GB |

## Notes
- Perplexity measured on wikitext-2.
- Memory measured during inference on 4 GPU.
- Calibration dataset: 1,024 tokens.
