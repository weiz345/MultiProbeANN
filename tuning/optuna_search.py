"""
Pareto optimization of multiprobe on ann-benchmarks HDF5 datasets.

Loads pre-split train/test/neighbors from standard ann-benchmarks HDF5 files
and runs NSGA-II to find Pareto-optimal (pca_dims, grid_size, n_probe) configs.
This is how the per-dataset multiprobe sweeps reported in the paper were derived;
feed the resulting Pareto front to extract_pareto_configs.py to produce
ann-benchmarks config.yml run_groups.

Objectives (both minimized):
  1. (1 - recall@K)           — maximize recall
  2. query_time / bf_time     — minimize relative latency

Usage:
    python tuning/optuna_search.py --config configs/glove200_search.yaml
"""

import argparse
import os
import time
from typing import Callable, Tuple

import h5py
import numpy as np
import optuna

import yaml

from ann.similarity.backends.bruteforce import BruteForceSimilarity
from ann.similarity.backends.multiprobe import MultiProbeSimilarity

METRIC_MAP = {"angular": "cosine", "euclidean": "euclidean"}


def load_hdf5(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ann-benchmarks HDF5 file.

    Returns
    -------
    train : (N, D) float32 — index vectors
    test  : (Q, D) float32 — query vectors
    neighbors : (Q, 100) int32 — precomputed ground-truth neighbor indices
    """
    with h5py.File(path, "r") as h5f:
        train = np.asarray(h5f["train"], dtype=np.float32)
        test = np.asarray(h5f["test"], dtype=np.float32)
        neighbors = np.asarray(h5f["neighbors"], dtype=np.int32)
    print(f"Loaded HDF5: train={train.shape}, test={test.shape}, neighbors={neighbors.shape}")
    return train, test, neighbors


def compute_recall_at_k(retrieved: list, ground_truth: np.ndarray, k: int) -> float:
    """Recall@K via set intersection: |retrieved[:k] ∩ gt[:k]| / k."""
    return len(set(retrieved[:k]) & set(ground_truth[:k].tolist())) / k


def measure_brute_force_time(
    vectors: np.ndarray,
    queries: np.ndarray,
    metric: str,
    k: int,
    n_sample: int = 50,
) -> float:
    """Average brute-force query time in ms over a sample of queries."""
    sim = BruteForceSimilarity(vectors, metric=metric)
    n = min(n_sample, len(queries))
    t0 = time.perf_counter()
    for q in queries[:n]:
        sim.query_vector(q, k=k)
    return ((time.perf_counter() - t0) / n) * 1000


def evaluate_config(
    vectors: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    metric: str,
    k: int,
    pca_dims: int,
    grid_size: int,
    n_probe: int,
) -> Tuple[float, float]:
    """Build multiprobe index and evaluate recall@K + avg query time.

    Returns (mean_recall, mean_time_ms).
    """
    sim = MultiProbeSimilarity(
        vectors, metric=metric, grid_dims=pca_dims, grid_splits=grid_size,
    )

    times = []
    total_recall = 0.0

    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        results = sim.query_vector(q, k=k, n_probe=n_probe)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        total_recall += compute_recall_at_k(list(results), ground_truth[i], k)

    return total_recall / len(queries), float(np.mean(times)) * 1000


def clamp_ranges(
    pca_dims: int,
    grid_size_range: Tuple[int, int],
    n_probe_range: Tuple[int, int],
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Clamp the search space for a given pca_dims.

    grid_size is capped so total cells (grid_size ** pca_dims) stay <= 1M, and
    n_probe is capped at 2^pca_dims (the quadrant neighbors actually probed).
    Returns ((grid_min, grid_max), (probe_min, probe_max)).
    """
    limit_grid = int(1_000_000 ** (1 / pca_dims))
    grid_max = min(grid_size_range[1], limit_grid)
    grid_min = min(grid_size_range[0], grid_max)

    max_probes = 2 ** pca_dims
    probe_max = min(n_probe_range[1], max_probes)
    probe_min = min(n_probe_range[0], probe_max)
    return (grid_min, grid_max), (probe_min, probe_max)


def make_objective(
    vectors: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    metric: str,
    recall_k: int,
    brute_force_time: float,
    pca_dim_range: Tuple[int, int],
    grid_size_range: Tuple[int, int],
    n_probe_range: Tuple[int, int],
) -> Callable:
    """Optuna NSGA-II objective factory."""

    def objective(trial: optuna.trial.Trial) -> Tuple[float, float]:
        pca_dims = trial.suggest_int("pca_dims", pca_dim_range[0], pca_dim_range[1])

        (grid_min, grid_max), (probe_min, probe_max) = clamp_ranges(
            pca_dims, grid_size_range, n_probe_range)
        grid_size = trial.suggest_int("grid_size", grid_min, grid_max)
        n_probe = trial.suggest_int("n_probe", probe_min, probe_max)

        try:
            recall, avg_time = evaluate_config(
                vectors, queries, ground_truth, metric, recall_k,
                pca_dims, grid_size, n_probe,
            )
        except Exception as e:
            print(f"[FAIL] Trial {trial.number}: {e}")
            raise optuna.TrialPruned() from e

        trial.set_user_attr("recall_at_k", recall)
        trial.set_user_attr("avg_time_ms", avg_time)
        trial.set_user_attr("speedup", brute_force_time / (avg_time + 1e-9))
        trial.set_user_attr("cells_total", grid_size ** pca_dims)

        print(
            f"Trial {trial.number}: Recall@{recall_k}={recall:.3f}, "
            f"Time={avg_time:.3f}ms "
            f"(d={pca_dims}, g={grid_size}, probe={n_probe})"
        )

        return 1.0 - recall, avg_time / brute_force_time

    return objective


def run(config_path: str) -> optuna.Study:
    """Main: load HDF5, run NSGA-II, print Pareto front."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Load data (hdf5_path may reference ${ANN_BENCHMARKS_DIR})
    train, test, neighbors = load_hdf5(os.path.expandvars(cfg["hdf5_path"]))

    # Map metric
    ann_metric = cfg["metric"]
    metric = METRIC_MAP[ann_metric]
    print(f"Metric: {ann_metric} -> {metric}")

    if metric == "cosine":
        train = train / (np.linalg.norm(train, axis=1, keepdims=True) + 1e-8)
        test = test / (np.linalg.norm(test, axis=1, keepdims=True) + 1e-8)

    # Optuna config
    opt = cfg["optuna"]
    recall_k = opt["recall_k"]
    pca_dim_range = tuple(opt["pca_dim_range"])
    grid_size_range = tuple(opt["grid_size_range"])
    n_probe_range = tuple(opt["n_probe_range"])
    n_trials = opt["n_trials"]
    study_name = opt["study_name"]
    storage = opt.get("storage")
    if storage:
        storage = os.path.expandvars(storage.format(study_name=study_name))
    bf_sample = opt.get("brute_force_sample", 50)
    seed = opt.get("seed", 42)

    # Brute-force baseline
    print(f"Measuring brute-force baseline ({bf_sample} queries)...")
    bf_time = measure_brute_force_time(train, test, metric, recall_k, bf_sample)
    print(f"  Brute-force avg query time: {bf_time:.3f} ms")

    # Build objective
    objective = make_objective(
        train, test, neighbors, metric, recall_k, bf_time,
        pca_dim_range, grid_size_range, n_probe_range,
    )

    # Ensure DB directory exists
    if storage and storage.startswith("sqlite:///"):
        db_dir = os.path.dirname(storage[len("sqlite:///"):])
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        directions=["minimize", "minimize"],
        load_if_exists=True,
        sampler=optuna.samplers.NSGAIISampler(seed=seed),
    )

    study.optimize(objective, n_trials=n_trials)

    # Report Pareto front
    print("\n" + "-" * 70)
    print("Pareto front:")
    best = sorted(study.best_trials, key=lambda t: t.user_attrs.get("recall_at_k", 0), reverse=True)

    header = f"{'Recall@'+str(recall_k):<12} {'Time(ms)':<10} {'Speedup':<10} | {'Params (d/g/probe)':<30}"
    print(header)
    print("-" * 70)

    for t in best:
        rec = t.user_attrs.get("recall_at_k", 0)
        time_ms = t.user_attrs.get("avg_time_ms", 0)
        speedup = t.user_attrs.get("speedup", 0)
        p = t.params
        param_str = f"d={p['pca_dims']}, g={p['grid_size']}, probe={p['n_probe']}"
        print(f"{rec:<12.3f} {time_ms:<10.3f} {speedup:<10.1f} | {param_str:<30}")

    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pareto optimization on ann-benchmarks HDF5")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    run(args.config)
