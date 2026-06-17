import numpy as np
import pytest

from ann_benchmarks_integration.module import AnnMultiprobe


@pytest.fixture
def corpus():
    return np.random.default_rng(42).random((400, 32)).astype(np.float32)


@pytest.mark.parametrize("metric, internal", [("angular", "cosine"), ("euclidean", "euclidean")])
def test_metric_mapping(metric, internal):
    algo = AnnMultiprobe(metric, grid_dims=3, grid_splits=6)
    assert algo._metric == internal


def test_invalid_metric_rejected():
    with pytest.raises(KeyError):
        AnnMultiprobe("hamming", grid_dims=3, grid_splits=6)


@pytest.mark.parametrize("metric", ["angular", "euclidean"])
def test_fit_then_query(corpus, metric):
    algo = AnnMultiprobe(metric, grid_dims=3, grid_splits=8)
    algo.fit(corpus)
    algo.set_query_arguments(4)
    res = algo.query(corpus[0], 10)
    assert isinstance(res, list)
    assert len(res) <= 10
    assert all(isinstance(r, (int, np.integer)) for r in res)


def test_name_reflects_n_probe(corpus):
    algo = AnnMultiprobe("angular", grid_dims=3, grid_splits=6)
    algo.fit(corpus)
    for n_probe in [1, 2, 8]:
        algo.set_query_arguments(n_probe)
        assert f"n_probe={n_probe}" in str(algo)


def test_recall_improves_with_probes(corpus):
    algo = AnnMultiprobe("angular", grid_dims=3, grid_splits=6)
    algo.fit(corpus)

    def mean_recall(n_probe, k=10, nq=20):
        algo.set_query_arguments(n_probe)
        recalls = []
        for i in range(nq):
            q = corpus[i]
            denom = np.linalg.norm(corpus, axis=1) * np.linalg.norm(q) + 1e-9
            gt = set(np.argsort((corpus @ q) / denom)[::-1][:k].tolist())
            recalls.append(len(set(algo.query(q, k)) & gt) / k)
        return float(np.mean(recalls))

    assert mean_recall(8) >= mean_recall(1) - 0.05
