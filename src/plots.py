"""make figures (mean and std across seeds) from results/results.json"""
import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


def agg(rows, method, f):
    """returns (prune % array, mean array, std array) of f(row) across seeds, per prune level"""
    by = defaultdict(list)
    for r in rows:
        if r["method"] == method:
            v = f(r)
            if v is not None:
                by[r["prune_amount"]].append(v)
    xs = sorted(by)
    return (np.array(xs) * 100, np.array([np.mean(by[x]) for x in xs]),
            np.array([np.std(by[x], ddof=1) if len(by[x]) > 1 else 0 for x in xs]))


def main(inp="results/results.json"):
    rows = json.load(open(inp))
    n = len({r["seed"] for r in rows})

    for m, label in [("prune", "pruning only (fp32)"), ("prune+int8", "pruning + int8")]:
        x, mu, sd = agg(rows, m, lambda r: r["accuracy"] * 100)
        plt.errorbar(x, mu, yerr=sd, marker="o", capsize=3, label=label)
    plt.xlabel("weights pruned (%)"); plt.ylabel("test accuracy (%)")
    plt.title(f"Accuracy vs. compression (mean ± std, n={n} seeds)")
    plt.grid(alpha=.3); plt.legend()
    plt.savefig("results/accuracy_vs_pruning.png", dpi=150); plt.close()

    for key, label in [("dense_fp32", "dense fp32"), ("sparse_fp32", "sparse fp32 (ideal)"),
                       ("zlib_fp32", "zlib fp32"), ("sparse_int8", "sparse int8 (ideal)"),
                       ("zlib_int8", "zlib int8")]:
        m = "prune+int8" if "int8" in key else "prune"
        x, mu, sd = agg(rows, m, lambda r: r["sizes"][key] / 1024)
        plt.errorbar(x, mu, yerr=sd, marker="o", capsize=3, label=label)
    plt.xlabel("weights pruned (%)"); plt.ylabel("size (KiB)")
    plt.title(f"Storage size by format (mean ± std, n={n} seeds)")
    plt.grid(alpha=.3); plt.legend()
    plt.savefig("results/size_vs_pruning.png", dpi=150); plt.close()

    print(f"{'prune%':>6} | {'prune only':>14} | {'prune+int8':>14}   (acc %, mean ± std)")
    a = agg(rows, "prune", lambda r: r["accuracy"] * 100)
    b = agg(rows, "prune+int8", lambda r: r["accuracy"] * 100)
    for i in range(len(a[0])):
        print(f"{a[0][i]:6.0f} | {a[1][i]:6.2f} ± {a[2][i]:4.2f} | {b[1][i]:6.2f} ± {b[2][i]:4.2f}")


if __name__ == "__main__":
    main()
