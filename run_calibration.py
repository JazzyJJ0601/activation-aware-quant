"""Run pipeline for calibration and quantization demo."""
from src.activation_aware_quant import ActivationAwareQuantizer
import numpy as np

class MockModel:
    """Mock model for demonstration."""
    pass

def main():
    q = ActivationAwareQuantizer()
    
    # 1. Calibration with mock data
    data = [np.random.randn(32, 64) for _ in range(5)]
    q.calibrate(MockModel(), data)
    print("Calibration stats:", q.layer_stats)
    
    # 2. Assign bit widths
    q.assign_bitwidths("percentile")
    print("Bit widths assigned:", q.layer_bitwidths)
    
    # 3. Quantize a sample tensor
    sample = np.random.randn(32, 64)
    q_sample = q.quantize(sample, "layer_0")
    print("Sample quantized successfully:", q_sample.shape)

if __name__ == "__main__":
    main()
