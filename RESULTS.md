# Real Results: Activation-Aware Quantization

**Status:** Measured on Qwen3-8B (3 short prompts): mixed precision 32.50 perplexity vs FP16 33.48 vs uniform 4-bit 35.15. Memory use has not been measured yet, and 3 prompts is an early signal, not a benchmark.

Command: `python3 repos/activation-aware-quant/results/run_real.py`

## Comparison Table

| Method | Avg Perplexity |
|--------|----------------|
| FP16 Baseline | 33.48 |
| Activation-Aware (Mixed) | 32.50 |
| Uniform 4-bit | 35.15 |

## Interpretation

The activation-aware mixed-precision method achieves a perplexity of 32.50,
comparable to the FP16 baseline (33.48) (memory not yet measured).
By assigning higher bit-widths to sensitive layers and lower bit-widths to insensitive layers,
the method preserves model accuracy better than uniform quantization (35.15 PPL).
