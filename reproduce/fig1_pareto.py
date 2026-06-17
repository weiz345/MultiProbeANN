"""
Fig 1: Pareto fronts on glove-200-angular (N=1.18M), all algorithms.

Reads ann-benchmarks result HDF5 files (set ANN_BENCHMARKS_DIR or pass
--ann-benchmarks-dir) and plots QPS vs recall@10 for each algorithm. See
docs/REPRODUCTION.md for how to generate the underlying runs.

Usage:
    python reproduce/fig1_pareto.py --ann-benchmarks-dir /path/to/ann-benchmarks
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from _common import (  # noqa: E402
    ALGO_LABELS,
    COLORS,
    build_pareto_front,
    load_results,
    resolve_ann_benchmarks_dir,
)

ALGO_ORDER = ["ann-multiprobe", "voyager", "pynndescent", "annoy", "faiss-ivf", "bruteforce"]
DATASET = "glove-200-angular"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ann-benchmarks-dir", default=None,
                        help="path to ann-benchmarks clone (default: $ANN_BENCHMARKS_DIR)")
    parser.add_argument("--output", default="fig1_pareto.png", help="output image path")
    args = parser.parse_args()

    ann_dir = resolve_ann_benchmarks_dir(args.ann_benchmarks_dir)
    all_results = load_results(ann_dir, DATASET)
    n_algos = len(set(r["algorithm"] for r in all_results))
    print(f"Loaded {len(all_results)} results across {n_algos} algorithms")

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    for algo in ALGO_ORDER:
        algo_data = [r for r in all_results if r["algorithm"] == algo]
        recalls, qps_vals = build_pareto_front(algo_data)
        if len(recalls) < 2:
            if algo_data:
                ax.plot([algo_data[0]["recall"]], [algo_data[0]["qps"]],
                        color=COLORS[algo], marker="o", markersize=6,
                        linewidth=0, label=ALGO_LABELS[algo])
            continue
        ax.plot(recalls, qps_vals, color=COLORS[algo], marker="o",
                markersize=4, linewidth=1.5, alpha=0.85, label=ALGO_LABELS[algo])

    ax.set_yscale("log")
    ax.set_xlabel("Recall@$k$=10", fontsize=16)
    ax.set_ylabel("Queries per second (1/s)", fontsize=16)
    ax.grid(True, alpha=0.2, which="both")
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
