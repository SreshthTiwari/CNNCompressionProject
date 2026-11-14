"""sparsity, storage size and latency measurements

zeroed weights do not shrink dense file or speed up dense kernels, so we report
several size notions separately:
  dense_fp32 : every weight as fp32 (what torch.save writes, regardless of zeros)
  dense_int8 : every weight as int8 (+ fp32 scales)
  sparse_* : idealized sparse storage = nonzeros * (value bytes + 4 byte index)
  zlib_* : dense bytes compressed with zlib (real, lossless file size)
"""
import time
import zlib
import numpy as np
import torch
from .model import weight_layers


def nonzero_fraction(model):
    layers = weight_layers(model)
    nz = sum((l.weight != 0).sum().item() for l in layers)
    tot = sum(l.weight.numel() for l in layers)
    return nz / tot


def _bias_bytes(model):
    return sum(l.bias.numel() * 4 for l in weight_layers(model) if l.bias is not None)


def storage_sizes(model, qweights=None):
    """sizes in bytes
    if `qweights` given, also report int8 variants"""
    layers = weight_layers(model)
    nz = sum((l.weight != 0).sum().item() for l in layers)
    n = sum(l.weight.numel() for l in layers)
    b = _bias_bytes(model)
    fp32_bytes = b"".join(l.weight.detach().numpy().astype(np.float32).tobytes() for l in layers)
    out = {
        "dense_fp32": n * 4 + b,
        "sparse_fp32": nz * (4 + 4) + b,
        "zlib_fp32": len(zlib.compress(fp32_bytes, 9)) + b,
    }
    if qweights is not None:
        scales = 4 * len(qweights)
        int8_bytes = b"".join(q.numpy().tobytes() for q, _ in qweights.values())
        out.update({
            "dense_int8": n + scales + b,
            "sparse_int8": nz * (1 + 4) + scales + b,
            "zlib_int8": len(zlib.compress(int8_bytes, 9)) + scales + b,
        })
    return out


@torch.no_grad()
def inference_time(model, loader, repeats=3):
    """mean seconds per test set pass on cpu (dense kernels; sparsity gives no speedup)"""
    model.eval().cpu()
    batches = [x for x, _ in loader]
    model(batches[0])
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        for x in batches:
            model(x)
        times.append(time.perf_counter() - t)
    return float(np.mean(times))
