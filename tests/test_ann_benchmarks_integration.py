"""Verify the ann-benchmarks integration wrapper works end-to-end."""

import sys
import os

import numpy as np

# Make the repo root and src/ importable so we can load the real AnnMultiprobe class
_repo_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _repo_root)
sys.path.insert(0, os.path.join(_repo_root, "src"))

from ann_benchmarks_integration.module import AnnMultiprobe


def _brute_force_knn(X, query, k, metric):
    """Return ground-truth k nearest neighbor indices via exhaustive search."""
    if metric == "euclidean":
        dists = np.linalg.norm(X - query, axis=1)
        return np.argsort(dists)[:k].tolist()
    else:  # cosine
        dots = X @ query
        norms = np.linalg.norm(X, axis=1) * np.linalg.norm(query) + 1e-9
        sims = dots / norms
        return np.argsort(sims)[::-1][:k].tolist()


def _recall(predicted, ground_truth):
    """Fraction of ground-truth neighbors found in predicted results."""
    return len(set(predicted) & set(ground_truth)) / len(ground_truth)


def _mean_recall(algo, X, k, metric, n_queries=20):
    """Average recall over multiple random queries."""
    rng = np.random.default_rng(99)
    indices = rng.choice(len(X), size=n_queries, replace=False)
    recalls = []
    for i in indices:
        results = algo.query(X[i], k)
        gt = _brute_force_knn(X, X[i], k, metric)
        recalls.append(_recall(results, gt))
    return float(np.mean(recalls))


def test_integration():
    rng = np.random.default_rng(42)
    X = rng.random((500, 64)).astype(np.float32)
    k = 10

    for metric_name, internal_metric in [("euclidean", "euclidean"), ("angular", "cosine")]:
        algo = AnnMultiprobe(metric_name, grid_dims=3, grid_splits=8)
        algo.fit(X)

        prev_recall = 0.0
        for n_probe in [1, 2, 4, 8]:
            algo.set_query_arguments(n_probe)

            # Basic contract checks on a single query
            results = algo.query(X[0], k)
            assert isinstance(results, list)
            assert len(results) <= k
            assert all(isinstance(r, int) for r in results)

            # self.name must reflect current n_probe
            assert f"n_probe={n_probe}" in str(algo)

            # Recall validation averaged over multiple queries
            avg_r = _mean_recall(algo, X, k, internal_metric)

            # More probes should not degrade average recall
            assert avg_r >= prev_recall - 0.05, (
                f"{metric_name}: n_probe={n_probe} recall ({avg_r:.2f}) "
                f"regressed vs prior ({prev_recall:.2f})"
            )
            prev_recall = avg_r

            print(
                f"{metric_name:10s} n_probe={n_probe:2d} -> {algo} "
                f"(avg recall={avg_r:.2f})"
            )

        # With enough probes, recall should be meaningfully above random
        # (random chance = k/N = 10/500 = 0.02)
        assert prev_recall >= 0.15, (
            f"{metric_name}: best recall {prev_recall:.2f} < 0.15 even at n_probe=8"
        )

    print("\nAll integration checks passed!")


if __name__ == "__main__":
    test_integration()
