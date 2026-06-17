import numpy as np
import pytest

from reproduce._common import (
    build_pareto_front,
    compute_recall,
    compute_scaling_exponents,
    interpolate_qps_at_recall,
    load_ground_truth,
    load_results,
)


def _neighbors_with_recall(gt, recall, k=10):
    m = round(recall * k)
    res = np.full((gt.shape[0], k), -1, dtype=np.int32)
    res[:, :m] = gt[:, :m]
    return res


def test_compute_recall_exact():
    gt = np.tile(np.arange(10), (5, 1)).astype(np.int32)
    res = _neighbors_with_recall(gt, 0.8)
    assert compute_recall(res, gt, k=10) == pytest.approx(0.8)


def test_build_pareto_front_keeps_upper_envelope():
    results = [
        {"recall": 0.2, "qps": 100.0},
        {"recall": 0.4, "qps": 80.0},   # dominated by (0.5, 90)
        {"recall": 0.5, "qps": 90.0},
        {"recall": 0.6, "qps": 50.0},
    ]
    recalls, qps = build_pareto_front(results)
    assert recalls.tolist() == [0.2, 0.5, 0.6]
    assert qps.tolist() == [100.0, 90.0, 50.0]


def test_build_pareto_front_empty():
    recalls, qps = build_pareto_front([])
    assert len(recalls) == 0 and len(qps) == 0


def test_interpolate_qps_log_linear():
    recalls = np.array([0.2, 0.6])
    qps = np.array([100.0, 1000.0])  # log10: 2.0, 3.0
    # midpoint recall -> log10 = 2.5 -> 10**2.5
    assert interpolate_qps_at_recall(recalls, qps, 0.4) == pytest.approx(10 ** 2.5, rel=1e-6)


@pytest.mark.parametrize("target", [0.1, 0.7])
def test_interpolate_out_of_range_returns_none(target):
    recalls = np.array([0.2, 0.6])
    qps = np.array([100.0, 1000.0])
    assert interpolate_qps_at_recall(recalls, qps, target) is None


def test_compute_scaling_exponents_recovers_power_law():
    alpha, c = -0.9, 1000.0

    def front(x):
        v = c * x ** alpha
        return [
            {"algorithm": "m", "recall": 0.5, "qps": 2 * v},
            {"algorithm": "m", "recall": 0.7, "qps": v},      # knot at the target
            {"algorithm": "m", "recall": 0.9, "qps": 0.5 * v},
        ]

    datasets = {"a": 1e4, "b": 1e5, "c": 1e6, "d": 1e7}
    cache = {name: front(x) for name, x in datasets.items()}
    targets, alphas = compute_scaling_exponents(datasets, "m", [0.7], cache)
    assert targets == [0.7]
    assert alphas[0] == pytest.approx(alpha, abs=1e-6)


def test_load_results_and_ground_truth(add_dataset):
    gt = np.tile(np.arange(10), (5, 1)).astype(np.int32)
    specs = [
        {"algo": "ann-multiprobe", "name": "cfg_a", "best_time": 0.01,
         "neighbors": _neighbors_with_recall(gt, 0.8)},
        {"algo": "ann-multiprobe", "name": "cfg_b", "best_time": 0.02,
         "neighbors": _neighbors_with_recall(gt, 0.9)},
        {"algo": "ann-multiprobe", "name": "broken", "best_time": 0.0,
         "neighbors": _neighbors_with_recall(gt, 1.0)},  # zero time -> skipped
    ]
    ann_dir = add_dataset("ds", gt, specs)

    assert load_ground_truth(ann_dir, "ds").shape == (5, 10)

    results = load_results(ann_dir, "ds")
    assert len(results) == 2  # broken file skipped
    by_qps = {round(r["qps"]): round(r["recall"], 2) for r in results}
    assert by_qps == {100: 0.8, 50: 0.9}


def test_load_results_missing_dataset_returns_empty(add_dataset):
    gt = np.tile(np.arange(10), (5, 1)).astype(np.int32)
    ann_dir = add_dataset("ds", gt, [])
    assert load_results(ann_dir, "does-not-exist") == []
