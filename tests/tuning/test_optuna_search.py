import numpy as np
import optuna
import pytest

from tuning.optuna_search import (
    clamp_ranges,
    compute_recall_at_k,
    evaluate_config,
    make_objective,
    measure_brute_force_time,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _unit_data(n=80, d=8, q=10, k=10, seed=1):
    rng = np.random.default_rng(seed)
    train = rng.standard_normal((n, d)).astype(np.float32)
    test = rng.standard_normal((q, d)).astype(np.float32)
    train /= np.linalg.norm(train, axis=1, keepdims=True) + 1e-8
    test /= np.linalg.norm(test, axis=1, keepdims=True) + 1e-8
    gt = np.array([np.argsort(train @ x)[::-1][:k] for x in test], dtype=np.int32)
    return train, test, gt


def test_compute_recall_at_k_set_intersection():
    retrieved = [0, 1, 2, 3, 4]
    gt = np.array([0, 1, 9, 8, 7], dtype=np.int32)
    assert compute_recall_at_k(retrieved, gt, k=5) == 0.4


def test_measure_brute_force_time_positive():
    train, test, _ = _unit_data()
    t = measure_brute_force_time(train, test, "cosine", k=10, n_sample=10)
    assert t > 0


def test_evaluate_config_single_cell_is_exact():
    train, test, gt = _unit_data()
    recall, time_ms = evaluate_config(train, test, gt, "cosine", k=10,
                                      pca_dims=2, grid_size=1, n_probe=1)
    assert recall == 1.0  # single cell gathers all points -> exact
    assert time_ms > 0


def test_evaluate_config_returns_valid_recall():
    train, test, gt = _unit_data()
    recall, _ = evaluate_config(train, test, gt, "cosine", k=10,
                                pca_dims=2, grid_size=4, n_probe=2)
    assert 0.0 <= recall <= 1.0


@pytest.mark.parametrize("dims", [2, 3, 5, 6, 8])
def test_clamp_ranges_caps_cells_and_probes(dims):
    # Ranges deliberately exceed both caps; the clamp must pull them back.
    (grid_min, grid_max), (probe_min, probe_max) = clamp_ranges(dims, (2, 2000), (1, 5000))
    assert grid_max ** dims <= 1_000_000
    assert grid_max == int(1_000_000 ** (1 / dims))
    assert probe_max == 2 ** dims
    assert grid_min <= grid_max and probe_min <= probe_max


def test_objective_runs_and_returns_valid_pair():
    train, test, gt = _unit_data()
    bf_time = measure_brute_force_time(train, test, "cosine", 10, 10)
    objective = make_objective(
        train, test, gt, "cosine", recall_k=10, brute_force_time=bf_time,
        pca_dim_range=(2, 3), grid_size_range=(2, 6), n_probe_range=(1, 8),
    )
    study = optuna.create_study(
        directions=["minimize", "minimize"],
        sampler=optuna.samplers.NSGAIISampler(seed=0),
    )
    study.optimize(objective, n_trials=6)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert completed
    for t in completed:
        assert t.params["n_probe"] <= 2 ** t.params["pca_dims"]
        assert len(t.values) == 2
        assert 0.0 <= t.values[0] <= 1.0   # 1 - recall
        assert t.values[1] >= 0.0          # time / bf_time
