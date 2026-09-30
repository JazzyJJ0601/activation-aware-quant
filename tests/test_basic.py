"""Basic smoke tests for activation-aware quantization."""

import pytest
import torch
import torch.nn as nn
from src.activation_aware_quant import ActivationAwareQuantizer


def test_quantizer_creation():
    """Test that quantizer can be instantiated."""
    quantizer = ActivationAwareQuantizer()
    assert quantizer.default_bitwidth == 8
    assert quantizer.magnitude_thresholds is not None


def test_quantize_function():
    """Test simulated quantization produces expected output."""
    quantizer = ActivationAwareQuantizer()
    
    # Test with simple tensor
    tensor = torch.tensor([0.0, 0.5, 1.0, 2.0, -1.0])
    quantized = quantizer.quantize(tensor, bitwidth=4)
    
    # Should have same shape
    assert quantized.shape == tensor.shape
    
    # Values should be close to original (quantization error within tolerance)
    assert torch.allclose(quantized, tensor, atol=0.15)


def test_calibration_and_bitwidth_assignment():
    """Test full calibration flow."""
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    quantizer = ActivationAwareQuantizer()
    
    # Create calibration data
    calibration_data = [torch.randn(32, 10) for _ in range(3)]
    
    # Run calibration
    magnitudes = quantizer.calibrate(model, calibration_data)
    
    # Should have some activation magnitudes captured
    assert len(magnitudes) > 0
    
    # Assign bit-widths based on magnitudes
    bitwidths = quantizer.assign_bitwidths(magnitudes)
    
    # Should have assigned bit-widths
    assert len(bitwidths) > 0


def test_quantization_pipeline():
    """Test end-to-end quantization pipeline."""
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.Linear(50, 10)
    )
    
    quantizer = ActivationAwareQuantizer()
    
    # Calibration
    calibration_data = [torch.randn(64, 100) for _ in range(5)]
    magnitudes = quantizer.calibrate(model, calibration_data)
    
    # Assign bit-widths
    bitwidths = quantizer.assign_bitwidths(magnitudes)
    
    # Quantize some weights
    for name, param in model.named_parameters():
        if 'weight' in name:
            bitwidth = quantizer.get_bitwidth_for_layer(name)
            quantized_weight = quantizer.quantize(param, bitwidth)
            
            # Should have same shape
            assert quantized_weight.shape == param.shape


def test_default_bitwidth():
    """Test that default bit-width is used for unassigned layers."""
    quantizer = ActivationAwareQuantizer()
    
    bitwidth = quantizer.get_bitwidth_for_layer("unknown_layer")
    assert bitwidth == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
