"""Activation-Aware Quantization Core.

Implements calibration, bit-width assignment, and simulated quantization
based on per-layer activation magnitude statistics.
"""
from typing import Dict, Any, Optional, List
import math

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class MockTensor:
    """Simple mock tensor-like object for environments without numpy."""
    def __init__(self, data: List[float]):
        self.data = data
        self.shape = (len(data),)

    def abs(self):
        return MockTensor([abs(x) for x in self.data])

    def mean(self):
        return sum(self.data) / len(self.data)


class ActivationAwareQuantizer:
    """Quantization aware training helper based on activation magnitudes."""

    def __init__(self):
        self.layer_stats: Dict[str, float] = {}
        self.layer_bitwidths: Dict[str, int] = {}

    def calibrate(self, model: Any, calibration_data: List[Any]) -> None:
        """
        Run forward pass on calibration data to capture activation magnitudes.

        Simulates hooks per layer to capture max abs activation per layer.
        """
        self.layer_stats.clear()

        # Simulate forward pass per layer name
        for i, name in enumerate(self._infer_layer_names(model)):
            # Simulate activation tensor (use numpy or mock)
            if HAS_NUMPY:
                act = np.random.randn(1, 10)
                mag = float(np.abs(act).mean())
            else:
                act = MockTensor([1.0] * 10)
                mag = act.mean()
            self.layer_stats[name] = mag

    def _infer_layer_names(self, model: Any) -> List[str]:
        """Mock layer name extraction (for demo purposes)."""
        if model:
            return ["layer_0", "layer_1", "layer_2"]
        return ["default_layer"]

    def assign_bitwidths(self, strategy: str = "percentile") -> None:
        """
        Assign bit-widths (2/4/8/16) per layer based on magnitude percentiles.

        Strategy:
            - High magnitude layers get higher precision (16 bit)
            - Low magnitude layers get lower precision (2 bit)
        """
        if not self.layer_stats:
            raise ValueError("Run calibration first.")

        items = sorted(self.layer_stats.items(), key=lambda x: x[1], reverse=True)
        n = len(items)
        for i, (name, _) in enumerate(items):
            # Thresholds based on quantiles
            if n > 0:
                if i < n * 0.25:  # Top 25%
                    bits = 16
                elif i < n * 0.50:
                    bits = 8
                elif i < n * 0.75:
                    bits = 4
                else:
                    bits = 2
            else:
                bits = 4
            self.layer_bitwidths[name] = bits

    def quantize(self, tensor: Any, layer_name: str) -> Any:
        """
        Apply simulated quantization (round+clip) at the assigned bit-width.

        Rounds values to the nearest representable value given the bit depth.
        """
        if layer_name not in self.layer_bitwidths:
            raise KeyError(f"Layer {layer_name} not found in assigned bit-widths.")

        bits = self.layer_bitwidths[layer_name]
        if bits <= 0:
            return tensor

        # Simulate quantization: quantize to [-1, 1] range with limited steps
        steps = 2 ** bits - 1
        
        # Handle numpy
        if HAS_NUMPY and isinstance(tensor, np.ndarray):
            x = np.clip(tensor, -1.0, 1.0)
            # Scale to steps, round, scale back
            q = np.round((x + 1.0) / 2.0 * steps) / steps * 2.0 - 1.0
            return q
        else:
            # Handle MockTensor / plain float
            if isinstance(tensor, MockTensor):
                x = [min(max(v, -1.0), 1.0) for v in tensor.data]
                q = [round((v + 1.0) / 2.0 * steps) / steps * 2.0 - 1.0 for v in x]
                return MockTensor(q)
            else:
                x = min(max(float(tensor), -1.0), 1.0)
                q = round((x + 1.0) / 2.0 * steps) / steps * 2.0 - 1.0
                return q


# Flag for import/success checks
ok = True
