# Activation-Aware Quantization

Low-bit weight quantisation that protects the input channels carrying large activations.
Before group-wise rounding, each input channel's weights are scaled up by `s = rms(x)^α`
and the scale is divided back out afterwards, so salient channels get finer steps at no
extra storage. α is searched per layer.

Measured on **Qwen3-8B**, same size as plain round-to-nearest (RTN):

| Bits | Packed size | Plain RTN | Activation-aware | bf16 |
|---:|---:|---:|---:|---:|
| 4 | 6.18 GB | 12.92 | **12.51** | 12.03 |
| 3 | 5.31 GB | 17.47 | **14.48** | 12.03 |

At 4 bits it removes 47% of RTN's perplexity loss; at 3 bits, 55%. The bf16 weights measure
16.38 GB, so 4-bit packing is 2.65× smaller and 3-bit 3.08×.

## Method

1. Run 8 × 512 tokens of WikiText-2 *train* and record E[x²] per input channel for all 252
   decoder linears.
2. For each layer and each α in {0, 0.1, …, 1}: scale columns by `s = rms(x)^α` (normalised so
   the geometric mean of max and min is 1), quantise with group-128 asymmetric RTN, unscale.
3. Keep the α with the lowest activation-weighted error Σ (ΔW)² · E[x²], a diagonal-Hessian proxy
   for output error. α = 0 is plain RTN, so the search can always fall back to it.

This is the core of AWQ (Lin et al., 2024) with a proxy-based α search per layer instead of
measuring block outputs. The scales fold into the preceding op (or into the dequant scale), so
the stored model has the same format and size as RTN.

The search picked a mean α of 0.35 and never chose α = 0: every one of the 252 layers preferred
some scaling.

## Honest limits

- Fake quantisation (quantise, then dequantise to bf16). Perplexity is exact for the format;
  packed size is computed (bits + fp16 scale and zero-point per 128 weights; embeddings, lm_head
  and norms stay 16-bit). No packed kernel, so no speed numbers.
- One model, WikiText-2 only, 40 × 512 evaluation tokens.
- Not compared with the official AWQ code or GPTQ; the comparison is against plain RTN in the
  same format.
- An earlier version claimed "mixed precision 32.50 beats FP16 33.48" on 3 short prompts. That
  came from a broken quantiser and too little text, and is withdrawn.

## Reproduce

Needs a CUDA GPU with ~20 GB and a local Qwen3-8B checkpoint (path in `results/qcommon.py`).

```bash
python results/run_real.py      # a few minutes on an RTX 3090 Ti, writes results/real.json
pytest -q tests
```

Details: [RESULTS.md](RESULTS.md). Raw numbers: [`results/real.json`](results/real.json).

## License

MIT
