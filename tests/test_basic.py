"""Tests for the quantiser and the activation-aware search used in results/run_real.py."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "results"))
from qcommon import rtn, weighted_err  # noqa: E402
from run_real import aware  # noqa: E402


def test_rtn_levels_and_error():
    torch.manual_seed(0)
    w = torch.randn(64, 256)
    for bits in (2, 3, 4):
        wq = rtn(w, bits)
        # each group of 128 uses at most 2**bits distinct values
        assert max(len(torch.unique(g)) for g in wq.reshape(-1, 128)) <= 2 ** bits
    e3 = (w - rtn(w, 3)).pow(2).mean()
    e4 = (w - rtn(w, 4)).pow(2).mean()
    assert e4 < e3 / 3


def test_rtn_exact_on_grid():
    w = torch.arange(128, dtype=torch.float32).repeat(4, 1) % 16
    assert torch.allclose(rtn(w, 4), w, atol=1e-5)


def test_aware_never_worse_than_rtn_on_proxy():
    torch.manual_seed(1)
    w = torch.randn(32, 256)
    act = torch.rand(256) ** 4 * 100  # a few channels with large activations
    wq, alpha = aware(w, 3, act)
    assert weighted_err(w, wq, act) <= weighted_err(w, rtn(w, 3), act) + 1e-6
    assert 0.0 <= alpha <= 1.0


def test_aware_helps_with_outlier_channels():
    torch.manual_seed(2)
    w = torch.randn(32, 256)
    act = torch.ones(256)
    act[:8] = 1e4
    wq, alpha = aware(w, 3, act)
    assert alpha > 0
    assert weighted_err(w, wq, act) < 0.8 * weighted_err(w, rtn(w, 3), act)
