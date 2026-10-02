#!/usr/bin/env python3
"""Real results: activation-aware weight scaling for low-bit quantisation of Qwen3-8B.

For each decoder linear, input channels that carry large activations are scaled up before
group-wise RTN (and the scale is divided back out), so their weights get finer steps:
  W_q = Q(W * s) / s,   s_j = rms(x_j) ** alpha
alpha is searched per layer (0, 0.1 .. 1) to minimise the activation-weighted weight error
(a diagonal-Hessian proxy for output error), using WikiText-2 *train* activations.
alpha = 0 is plain RTN, so the search can only pick RTN or something the proxy prefers.

Compared against plain group-wise RTN (the standard baseline) at 3 and 4 bits, group 128.
Perplexity on WikiText-2 *test*, 40 x 512 tokens. Packed weight size is computed
(bits + fp16 scale/zero per group); the fp16 footprint is measured with torch.
"""
import gc
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from qcommon import (act_stats, decoder_linears, load_model, packed_gb, perplexity, rtn,  # noqa: E402
                     save, weighted_err, windows)

OUT = HERE / "real.json"
N_TEST, N_CALIB = 40, 8
ALPHAS = [i / 10 for i in range(11)]


@torch.no_grad()
def aware(w, bits, act):
    rms = act.sqrt().clamp_min(1e-6)
    best = None
    for a in ALPHAS:
        s = rms ** a
        s = s / (s.max() * s.min()).sqrt()
        wq = rtn(w.float() * s.view(1, -1), bits) / s.view(1, -1)
        err = weighted_err(w, wq, act)
        if best is None or err < best[0]:
            best = (err, a, wq)
    return best[2], best[1]


def main():
    model, tok = load_model()
    test, calib = windows(tok, "test", N_TEST), windows(tok, "train", N_CALIB)
    linears = decoder_linears(model)
    originals = {n: m.weight.data.to("cpu", copy=True) for n, m in linears}
    acts = act_stats(model, linears, calib)
    results = {"setup": {"model": "Qwen3-8B", "group": 128, "eval": f"WikiText-2 test, {N_TEST}x512 tokens",
                         "calib": f"WikiText-2 train, {N_CALIB}x512 tokens", "linears": len(linears)}}

    torch.cuda.synchronize()
    results["fp16"] = {"ppl": round(perplexity(model, test), 4),
                       "measured_weights_gb": round(sum(p.numel() * p.element_size()
                                                        for p in model.parameters()) / 1e9, 2)}
    save(OUT, results)
    print("fp16", results["fp16"], flush=True)

    for bits in (4, 3):
        for method in ("rtn", "activation_aware"):
            t0, alphas = time.time(), []
            for n, m in linears:
                w = originals[n].cuda()
                if method == "rtn":
                    m.weight.data = rtn(w, bits).to(torch.bfloat16)
                else:
                    wq, a = aware(w, bits, acts[n])
                    m.weight.data = wq.to(torch.bfloat16)
                    alphas.append(a)
            row = {"ppl": round(perplexity(model, test), 4),
                   "packed_gb": packed_gb(model, {n: bits for n, _ in linears}),
                   "seconds": round(time.time() - t0, 1)}
            if alphas:
                row["mean_alpha"] = round(sum(alphas) / len(alphas), 3)
                row["layers_alpha0"] = alphas.count(0.0)
            results[f"{method}_{bits}bit"] = row
            save(OUT, results)
            print(f"{method}_{bits}bit", row, flush=True)
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
