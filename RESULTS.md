# Activation-Aware Quantization: results

Model: Qwen3-8B (bf16). All 252 decoder linears quantised; group 128, asymmetric RTN.
Calibration: WikiText-2 train, 8 × 512 tokens. Evaluation: WikiText-2 test, 40 × 512 tokens.
Command: `python results/run_real.py` (raw output in `results/real.json`).

| Config | Packed size | Perplexity | Quantise time | Mean α |
|---|---:|---:|---:|---:|
| bf16 | 16.38 GB (measured) | 12.0346 | – | – |
| RTN 4-bit | 6.18 GB | 12.9164 | 10.5 s | – |
| **Activation-aware 4-bit** | 6.18 GB | **12.5055** | 22.7 s | 0.348 |
| RTN 3-bit | 5.31 GB | 17.4711 | 10.4 s | – |
| **Activation-aware 3-bit** | 5.31 GB | **14.4764** | 22.8 s | 0.355 |

- 4-bit: loss vs bf16 falls from +0.88 to +0.47 perplexity.
- 3-bit: loss falls from +5.44 to +2.44.
- No layer chose α = 0 (plain RTN) at either bit-width.
- Quantising the whole model takes about 23 s on an RTX 3090 Ti (11 α values per layer).

Packed sizes are computed, not measured: quantised weights at `bits + 32/128` bits each, all other
parameters at 16 bits.

**Withdrawn.** The previous version of this file reported FP16 33.48 / mixed 32.50 / uniform 4-bit
35.15 on 3 short prompts. The quantiser behind those numbers used one scale per weight matrix
and was wrong; the numbers are replaced by the table above.
