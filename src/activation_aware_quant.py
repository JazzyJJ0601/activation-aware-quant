"""Activation-aware mixed-precision quantization module."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


class ActivationAwareQuantizer:
    """
    Quantizes model layers based on activation magnitudes observed during calibration.
    
    This quantizer captures activation statistics from a forward pass on calibration data,
    assigns appropriate bit-widths per layer based on activation magnitude thresholds,
    and applies simulated quantization to weight parameters.
    """
    
    def __init__(self, default_bitwidth: int = 8, 
                 magnitude_thresholds: Optional[Dict[int, float]] = None):
        """
        Initialize the quantizer.
        
        Args:
            default_bitwidth: Default bit-width for layers without special assignment.
            magnitude_thresholds: Dict mapping bit-width to magnitude threshold.
                Layers with activation magnitude above threshold get that bit-width.
        """
        self.default_bitwidth = default_bitwidth
        self.magnitude_thresholds = magnitude_thresholds or {
            16: 100.0,  # Very high magnitude -> 16-bit
            8: 10.0,    # High magnitude -> 8-bit
            4: 1.0,     # Medium magnitude -> 4-bit
        }
        self._activation_stats: Dict[str, torch.Tensor] = {}
        self._bitwidths: Dict[str, int] = {}
        
    def _register_hooks(self, module: nn.Module) -> None:
        """Register hooks to capture activation magnitudes."""
        def hook_fn(module, inputs, output):
            # Capture output activation magnitude
            if isinstance(output, torch.Tensor):
                key = f"{module.__class__.__name__}"
                if key not in self._activation_stats:
                    self._activation_stats[key] = []
                mag = output.abs().mean().item()
                self._activation_stats[key].append(mag)
        
        # Register forward hook on the module
        self._hook_handle = module.register_forward_hook(hook_fn)
    
    def _cleanup_hooks(self) -> None:
        """Remove registered hooks."""
        if hasattr(self, '_hook_handle'):
            self._hook_handle.remove()
    
    def calibrate(self, model: nn.Module, 
                  calibration_data: List[torch.Tensor]) -> Dict[str, float]:
        """
        Run forward pass on calibration data to collect activation statistics.
        
        Args:
            model: The model to calibrate.
            calibration_data: List of input tensors for calibration.
            
        Returns:
            Dict mapping layer names to their mean activation magnitudes.
        """
        # Clear previous stats
        self._activation_stats.clear()
        
        # Register hooks on all layers with weights
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding)):
                handle = module.register_forward_hook(
                    lambda m, inp, out, n=name: self._capture_activation(m, out)
                )
                hooks.append(handle)
        
        # Run calibration forward passes
        model.eval()
        with torch.no_grad():
            for input_tensor in calibration_data:
                model(input_tensor)
        
        # Compute mean activation magnitudes per layer
        mean_magnitudes = {}
        for name, magnitudes in self._activation_stats.items():
            if magnitudes:
                mean_magnitudes[name] = sum(magnitudes) / len(magnitudes)
        
        # Cleanup hooks
        for handle in hooks:
            handle.remove()
        
        return mean_magnitudes
    
    def _capture_activation(self, module: nn.Module, 
                            output: torch.Tensor) -> None:
        """Capture activation magnitude for a single layer."""
        if isinstance(output, torch.Tensor):
            key = f"{module.__class__.__name__}_{id(module)}"
            if key not in self._activation_stats:
                self._activation_stats[key] = []
            mag = output.abs().mean().item()
            self._activation_stats[key].append(mag)
    
    def assign_bitwidths(self, 
                         activation_magnitudes: Dict[str, float]) -> Dict[str, int]:
        """
        Assign bit-widths to layers based on activation magnitudes.
        
        Args:
            activation_magnitudes: Dict mapping layer names to mean activation magnitudes.
            
        Returns:
            Dict mapping layer names to assigned bit-widths.
        """
        self._bitwidths.clear()
        
        for layer_name, mag in activation_magnitudes.items():
            assigned = self.default_bitwidth
            # Assign higher bit-width for higher magnitudes
            for bitwidth, threshold in sorted(self.magnitude_thresholds.items(), 
                                             key=lambda x: -x[1]):
                if mag >= threshold:
                    assigned = bitwidth
                    break
            self._bitwidths[layer_name] = assigned
        
        return self._bitwidths.copy()
    
    def quantize(self, tensor: torch.Tensor, bitwidth: int) -> torch.Tensor:
        """
        Apply simulated quantization to a tensor.
        
        Args:
            tensor: The tensor to quantize.
            bitwidth: Number of bits for quantization.
            
        Returns:
            Quantized tensor (simulated via rounding and scaling).
        """
        # Compute quantization parameters
        max_val = tensor.abs().max().item()
        if max_val == 0:
            return tensor
        
        # Number of levels
        levels = 2 ** bitwidth
        
        # Scale tensor to [0, levels-1] range
        scale = max_val / (levels / 2)
        if scale == 0:
            return tensor
        
        # Quantize: scale -> round -> de-scale
        q_min = -max_val
        q_max = max_val
        q_range = q_max - q_min
        q_levels = levels
        
        # Clamp and scale
        scaled = (tensor - q_min) / q_range * (q_levels - 1)
        
        # Round to integer
        quantized_int = torch.round(scaled).clamp(0, q_levels - 1)
        
        # De-scale to original range
        quantized_tensor = quantized_int / (q_levels - 1) * q_range + q_min
        
        return quantized_tensor
    
    def get_bitwidth_for_layer(self, layer_name: str) -> int:
        """Get the assigned bit-width for a specific layer."""
        return self._bitwidths.get(layer_name, self.default_bitwidth)
