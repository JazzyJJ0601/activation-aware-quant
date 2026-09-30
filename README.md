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
