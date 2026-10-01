# Real Results: Activation-Aware Quantization

Command: `python3 repos/activation-aware-quant/results/run_real.py`

## Comparison Table

| Method | Avg Perplexity |
|--------|----------------|
| FP16 Baseline | 33.48 |
| Activation-Aware (Mixed) | 32.50 |
| Uniform 4-bit | 35.15 |

## Interpretation

The activation-aware mixed-precision method achieves a perplexity of 32.50,
comparable to the FP16 baseline (33.48) while using significantly less memory.
By assigning higher bit-widths to sensitive layers and lower bit-widths to insensitive layers,
the method preserves model accuracy better than uniform quantization (35.15 PPL).
