# Activation-Aware Quantization

This repository implements activation-aware mixed-precision quantization for large language models.

## What It Is
Activation-aware quantization dynamically assigns different bit-widths to layers based on their activation statistics, rather than treating all weights uniformly.

## Innovation
Traditional quantization methods rely on weight gradients or uniform bit-width assignment. Our approach uses activation statistics (outliers, variance, distribution) to identify which layers are sensitive to quantization and which can be aggressively compressed.

## How It Works
1. **Calibrate:** Run a forward pass on a small calibration dataset to collect activation statistics per layer.
2. **Assign Bit-Widths:** Use activation variance and outlier counts to assign optimal bit-widths (e.g., 4-bit, 6-bit, 8-bit) per layer.
3. **Apply Simulated Quantization:** Simulate quantization effects during training or inference to validate performance before deployment.

## Design Methodology (from repo9 design phase)
- **Activation Profiling:** Run a small calibration set (128 samples from WikiText-2) through the model. Record per-layer activation statistics (mean absolute value, variance, and outlier counts).
- **Sensitivity Scoring:** Compute a sensitivity score per layer based on activation stability. High variance or large magnitude activations indicate importance.
- **Bit-Width Assignment:**
  - **High Sensitivity:** FP16 or 4-bit to preserve gradient flow.
  - **Low Sensitivity:** 2-bit quantization for maximum compression.
- **Reconstruction Loss:** Fine-tune quantization parameters using mean squared error between quantized and original activations.

## Expected Memory Savings
Assuming Qwen3-8B has ~80 layers:
- 20% sensitive layers @ 4-bit
- 80% insensitive layers @ 2-bit
- Baseline 4-bit uniform: ~4 GB
- Proposed mixed precision: ~2.8 GB
- **Savings:** ~30% reduction vs. uniform 4-bit, with negligible perplexity cost.

## Planned Benchmark Methodology
- **Dataset:** WikiText-2 (validation split).
- **Metric:** Perplexity (PPL) and inference latency.
- **Baselines:**
  1. Full precision (FP16).
  2. Uniform 4-bit (baseline).
  3. Uniform 2-bit (compression extreme).
- **Evaluation:**
  1. Measure PPL for all configurations.
  2. Compare parameter memory footprint.
  3. Plot PPL vs. model size to identify optimal compression frontier.

## Expected Benefits
- **Lower Perplexity:** Maintains accuracy at lower bit-widths compared to uniform quantization.
- **Memory Efficiency:** Reduces model size by compressing less sensitive layers.
- **Hardware Friendly:** Uses integer arithmetic for compressed layers.

## Usage Example
```python
from quantization import ActivationAwareQuantizer

quantizer = ActivationAwareQuantizer(calibration_data=dataloader)
quantizer.calibrate(model)
quantizer.apply_quantization(model)
```

Or via CLI:
```bash
python quantize.py --model llama2-7b --calib calib.json --output quantized/model
```


## Results

See [RESULTS.md](RESULTS.md)
