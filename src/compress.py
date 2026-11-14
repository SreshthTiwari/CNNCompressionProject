"""pruning and quantization

pruning: global magnitude pruning (zero smallest |w| across all conv/linear weights)
quantization: symmetric per tensor int8 *simulated* (fake quant) weight quantization
  weights are rounded to 255 levels and stored as int8 + one fp32 scale per tensor;
  for accuracy evaluation they are dequantized back to float
  this isolates effect of reduced precision from backend specific int8 kernels (see readme)
"""
import copy
import torch
from .model import weight_layers


def prune_global(model, amount):
    """return copy with `amount` fraction of smallest magnitude weights zeroed"""
    m = copy.deepcopy(model)
    layers = weight_layers(m)
    if amount <= 0:
        return m
    all_w = torch.cat([l.weight.detach().abs().flatten() for l in layers])
    k = int(amount * all_w.numel())
    thresh = all_w.kthvalue(k).values
    with torch.no_grad():
        for l in layers:
            l.weight.mul_((l.weight.abs() > thresh).float())
    return m


def quantize_tensor(w, bits=8):
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().max().clamp(min=1e-12) / qmax
    q = torch.round(w / scale).clamp(-qmax, qmax).to(torch.int8)
    return q, scale


def quantize_model(model, bits=8):
    """return (dequantized copy for evaluation, {layer_idx: (int8 tensor, scale)})

    zeros stay exactly zero (round(0)=0), so pruning sparsity is preserved
    """
    m = copy.deepcopy(model)
    qweights = {}
    with torch.no_grad():
        for i, l in enumerate(weight_layers(m)):
            q, s = quantize_tensor(l.weight, bits)
            qweights[i] = (q, s)
            l.weight.copy_(q.float() * s)
    return m, qweights
