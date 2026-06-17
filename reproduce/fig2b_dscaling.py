"""
Fig 2b: d-scaling exponent (alpha_d) vs recall@10 on the GloVe family.

Across glove-25/50/100/200-angular (all N=1.18M, only dimensionality d varies),
fits the slope of log10(QPS) vs log10(d) at each recall target. Multiprobe Grid
stays relatively flat while graph/tree/partitioning methods steepen at high recall
(the d-scaling crossover). Set ANN_BENCHMARKS_DIR or pass --ann-benchmarks-dir.

Usage:
    python reproduce/fig2b_dscaling.py --ann-benchmarks-dir /path/to/ann-benchmarks
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from _common import (  # noqa: E402
    ALGO_LABELS,
    COLORS,
    compute_scaling_exponents,
    load_results,
    resolve_ann_benchmarks_dir,
)

ALGO_ORDER = ["ann-multiprobe", "voyager", "pynndescent", "annoy", "faiss-ivf"]

D_SCALING_DATASETS = {
    "glove-25-angular": 25,
    "glove-50-angular": 50,
    "glove-100-angular": 100,
    "glove-200-angular": 200,
}

RECALL_TARGETS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ann-benchmarks-dir", default=None,
                        help="path to ann-benchmarks clone (default: $ANN_BENCHMARKS_DIR)")
    parser.add_argument("--output", default="fig2b_dscaling.png", help="output image path")
    args = parser.parse_args()

    ann_dir = resolve_ann_benchmarks_dir(args.ann_benchmarks_dir)

    cache = {}
    for ds_name in D_SCALING_DATASETS:
        cache[ds_name] = load_results(ann_dir, ds_name)
        print(f"  {ds_name}: {len(cache[ds_name])} results")

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for algo in ALGO_ORDER:
        targets, alphas = compute_scaling_exponents(
            D_SCALING_DATASETS, algo, RECALL_TARGETS, cache)
        if len(targets) >= 2:
            ax.plot(targets, alphas, color=COLORS[algo], marker="o",
                    markersize=6, linewidth=2, label=ALGO_LABELS[algo])

    ax.set_xlabel("Recall@10 target", fontsize=16)
    ax.set_ylabel("$d$-scaling exponent ($\\alpha_d$)", fontsize=16)
    ax.set_title("$d$-scaling", fontsize=18)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=9, loc="lower left")

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
