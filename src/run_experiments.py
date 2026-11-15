"""train one baseline per seed, sweep pruning x quantization, write results/results.json"""
import argparse
import json
import os
import torch
from .data import get_loaders
from .model import SmallCNN
from .train import accuracy, train
from .compress import prune_global, quantize_model
from .metrics import nonzero_fraction, storage_sizes, inference_time

PRUNE_LEVELS = [0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def main(seeds=(0, 1, 2, 3, 4), epochs=10, out="results/results.json"):
    _, test_loader = get_loaders()
    rows = []
    for seed in seeds:
        ckpt = f"results/baseline_seed{seed}.pt"
        if not os.path.exists(ckpt):
            train(epochs=epochs, seed=seed, out=ckpt)
        base = SmallCNN()
        base.load_state_dict(torch.load(ckpt))
        for p in PRUNE_LEVELS:
            pruned = prune_global(base, p)
            qmodel, qw = quantize_model(pruned, bits=8)
            for name, model, q in [("prune", pruned, None), ("prune+int8", qmodel, qw)]:
                rows.append({
                    "seed": seed, "method": name, "prune_amount": p,
                    "accuracy": accuracy(model, test_loader),
                    "nonzero_fraction": nonzero_fraction(model),
                    "sizes": storage_sizes(model, q),
                    "inference_s": inference_time(model, test_loader),
                })
                print(f"seed={seed} {name:11s} p={p:.2f} acc={rows[-1]['accuracy']:.4f}")
    json.dump(rows, open(out, "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=10)
    a = ap.parse_args()
    main(tuple(a.seeds), a.epochs)
