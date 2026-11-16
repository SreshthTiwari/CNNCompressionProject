"""per layer sparsity under global magnitude pruning

global pruning ranks weights across all layers together, so layer that holds most of
weights absorbs most of pruning
this reports, per layer, fraction of weights that
remain nonzero at each global prune level (mean over seeds) saved to results/layer_sparsity.json
and results/layer_sparsity.png
"""
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from .model import SmallCNN, weight_layers
from .compress import prune_global
from .run_experiments import PRUNE_LEVELS

NAMES = ["conv1 (1->16)", "conv2 (16->32)", "fc1 (1568->128)", "fc2 (128->10)"]


def main(seeds=(0, 1, 2, 3, 4), out="results/layer_sparsity.json"):
    res = {}
    sizes = None
    for seed in seeds:
        base = SmallCNN()
        base.load_state_dict(torch.load(f"results/baseline_seed{seed}.pt"))
        sizes = [l.weight.numel() for l in weight_layers(base)]
        for p in PRUNE_LEVELS:
            layers = weight_layers(prune_global(base, p))
            res.setdefault(p, []).append([(l.weight != 0).float().mean().item() for l in layers])

    mean = {p: np.mean(v, axis=0) for p, v in res.items()}
    std = {p: np.std(v, axis=0, ddof=1) for p, v in res.items()}
    json.dump({"layers": NAMES, "n_weights": sizes,
               "nonzero_mean": {str(p): m.tolist() for p, m in mean.items()},
               "nonzero_std": {str(p): s.tolist() for p, s in std.items()}},
              open(out, "w"), indent=2)

    print("weights per layer:", dict(zip(NAMES, sizes)),
          f"({100 * sizes[2] / sum(sizes):.1f}% in fc1)")
    print(f"{'pruned':>6} | " + " | ".join(f"{n:>15}" for n in NAMES) + "   (% weights remaining)")
    for p in PRUNE_LEVELS:
        print(f"{p * 100:5.0f}% | " + " | ".join(f"{100 * v:15.1f}" for v in mean[p]))

    xs = [p * 100 for p in PRUNE_LEVELS]
    for i, n in enumerate(NAMES):
        plt.errorbar(xs, [100 * mean[p][i] for p in PRUNE_LEVELS],
                     yerr=[100 * std[p][i] for p in PRUNE_LEVELS], marker="o", capsize=3, label=n)
    plt.plot([0, 100], [100, 0], "k:", alpha=.4, label="uniform pruning")
    plt.xlabel("global weights pruned (%)"); plt.ylabel("weights remaining in layer (%)")
    plt.title(f"Per-layer sparsity under global pruning (n={len(seeds)} seeds)")
    plt.grid(alpha=.3); plt.legend()
    plt.savefig("results/layer_sparsity.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
