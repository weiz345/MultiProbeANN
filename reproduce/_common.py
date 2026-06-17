"""
Shared helpers for reproducing paper figures from ann-benchmarks result files.

The figure scripts read raw ann-benchmarks output:
    <ANN_BENCHMARKS_DIR>/results/<dataset>/10/<algo>/*.hdf5   (per-config runs)
    <ANN_BENCHMARKS_DIR>/data/<dataset>.hdf5                  (ground-truth neighbors)

Recall@k is recomputed against the dataset ground truth, the Pareto front is the
upper QPS envelope over configs, and scaling exponents are slopes of log10(QPS)
vs log10(x). These are the same primitives used to produce the paper figures.
"""

import os

import h5py
import numpy as np
from scipy import interpolate as sci_interp

ALGO_LABELS = {
    "ann-multiprobe": "Multiprobe Grid",
    "voyager": "Voyager (graph)",
    "pynndescent": "PyNNDescent (graph)",
    "annoy": "Annoy (tree)",
    "faiss-ivf": "FAISS-IVF (partitioning)",
    "bruteforce": "BruteForce (exact)",
}

COLORS = {
    "ann-multiprobe": "#000000",
    "voyager": "#007ac2",
    "pynndescent": "#cf4781",
    "annoy": "#e6a817",
    "faiss-ivf": "#2ca02c",
    "bruteforce": "#999999",
}


def resolve_ann_benchmarks_dir(cli_arg=None):
    """Return the ann-benchmarks clone path from --ann-benchmarks-dir or env."""
    path = cli_arg or os.environ.get("ANN_BENCHMARKS_DIR")
    if not path:
        raise SystemExit(
            "set ANN_BENCHMARKS_DIR (or pass --ann-benchmarks-dir) to your "
            "ann-benchmarks clone"
        )
    return path


def load_ground_truth(ann_dir, dataset_name):
    ds_path = os.path.join(ann_dir, "data", f"{dataset_name}.hdf5")
    with h5py.File(ds_path, "r") as f:
        return np.array(f["neighbors"], dtype=np.int32)


def compute_recall(result_neighbors, gt_neighbors, k=10):
    n_queries = min(len(result_neighbors), len(gt_neighbors))
    recalls = []
    for i in range(n_queries):
        retrieved = set(result_neighbors[i][:k].tolist())
        relevant = set(gt_neighbors[i][:k].tolist())
        recalls.append(len(retrieved & relevant) / k)
    return float(np.mean(recalls))


def load_results(ann_dir, dataset_name, k=10):
    """Load (algorithm, recall, qps) for every config run of a dataset."""
    results = []
    dataset_dir = os.path.join(ann_dir, "results", dataset_name, "10")
    if not os.path.exists(dataset_dir):
        return results
    gt = load_ground_truth(ann_dir, dataset_name)
    for algo_dir in os.listdir(dataset_dir):
        algo_path = os.path.join(dataset_dir, algo_dir)
        if not os.path.isdir(algo_path):
            continue
        for fname in os.listdir(algo_path):
            if not fname.endswith(".hdf5"):
                continue
            fpath = os.path.join(algo_path, fname)
            try:
                with h5py.File(fpath, "r") as h:
                    best_time = float(h.attrs.get("best_search_time", 0))
                    if best_time <= 0 or "neighbors" not in h:
                        continue
                    result_neighbors = np.array(h["neighbors"], dtype=np.int32)
                    recall = compute_recall(result_neighbors, gt, k=k)
                    results.append({
                        "algorithm": algo_dir,
                        "recall": recall,
                        "qps": 1.0 / best_time,
                    })
            except Exception:
                continue
    return results


def build_pareto_front(results):
    """Upper QPS envelope: highest QPS achievable at or above each recall."""
    if not results:
        return np.array([]), np.array([])
    points = sorted(results, key=lambda r: r["recall"])
    recalls, qps_vals = [], []
    best_qps = -1
    for p in reversed(points):
        if p["qps"] > best_qps:
            recalls.append(p["recall"])
            qps_vals.append(p["qps"])
            best_qps = p["qps"]
    recalls.reverse()
    qps_vals.reverse()
    return np.array(recalls), np.array(qps_vals)


def interpolate_qps_at_recall(recalls, qps_vals, target):
    """Linear interpolation of log10(QPS) at a target recall; None if out of range."""
    if len(recalls) < 2 or target < recalls.min() or target > recalls.max():
        return None
    log_qps = np.log10(qps_vals)
    f = sci_interp.interp1d(recalls, log_qps, kind="linear", fill_value="extrapolate")
    return 10 ** float(f(target))


def compute_scaling_exponents(datasets_dict, algo, recall_targets, load_cache):
    """Slope of log10(QPS) vs log10(x) at each recall target for one algorithm.

    datasets_dict maps dataset_name -> x value (N or d). load_cache maps
    dataset_name -> list of result dicts (from load_results).
    """
    targets_used = []
    alphas = []
    for target in recall_targets:
        xs, qps_list = [], []
        for ds_name, x_val in sorted(datasets_dict.items(), key=lambda kv: kv[1]):
            results = load_cache[ds_name]
            algo_data = [r for r in results if r["algorithm"] == algo]
            recalls, qps_vals = build_pareto_front(algo_data)
            qps = interpolate_qps_at_recall(recalls, qps_vals, target)
            if qps is not None:
                xs.append(x_val)
                qps_list.append(qps)
        if len(xs) >= 3:
            log_x = np.log10(xs)
            log_qps = np.log10(qps_list)
            coeffs = np.polyfit(log_x, log_qps, 1)
            targets_used.append(target)
            alphas.append(coeffs[0])
    return targets_used, alphas
