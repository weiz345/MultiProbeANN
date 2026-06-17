import numpy as np
import pytest

from ann.similarity.backends.bruteforce import BruteForceSimilarity
from ann.similarity.backends.multiprobe import MultiProbeSimilarity


def _true_neighbors(corpus, queries, k, metric):
    gt = []
    for q in queries:
        if metric == "cosine":
            denom = np.linalg.norm(corpus, axis=1) * np.linalg.norm(q) + 1e-9
            scores = (corpus @ q) / denom
            gt.append(set(np.argsort(scores)[::-1][:k].tolist()))
        else:
            d = np.linalg.norm(corpus - q, axis=1)
            gt.append(set(np.argsort(d)[:k].tolist()))
    return gt


def _mean_recall(engine, corpus, gt, k, n_probe, n_queries=25):
    recalls = []
    for i in range(n_queries):
        res = set(engine.query_vector(corpus[i], k=k, n_probe=n_probe))
        recalls.append(len(res & gt[i]) / k)
    return float(np.mean(recalls))


@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
def test_build_and_query_shape(synthetic_corpus, metric):
    eng = MultiProbeSimilarity(synthetic_corpus, metric=metric, grid_dims=3, grid_splits=6)
    res = eng.query_vector(synthetic_corpus[0], k=10, n_probe=4)
    assert isinstance(res, list)
    assert len(res) <= 10
    assert all(isinstance(r, (int, np.integer)) for r in res)


def test_grid_splits_accepts_int_and_tuple(synthetic_corpus):
    a = MultiProbeSimilarity(synthetic_corpus, metric="cosine", grid_dims=3, grid_splits=5)
    b = MultiProbeSimilarity(synthetic_corpus, metric="cosine", grid_dims=3, grid_splits=(5, 5, 5))
    assert a.grid.splits == b.grid.splits == (5, 5, 5)


def test_n_probe_one_visits_only_primary_cell(synthetic_corpus):
    eng = MultiProbeSimilarity(synthetic_corpus, metric="cosine", grid_dims=3, grid_splits=6)
    pt = (synthetic_corpus[0] - eng.pca_mean) @ eng.pca_comp
    primary = eng.grid._cell_index(pt)
    assert eng._get_nearest_cells(pt, primary, 1) == [primary]


@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
def test_recall_monotonic_in_n_probe(synthetic_corpus, metric):
    eng = MultiProbeSimilarity(synthetic_corpus, metric=metric, grid_dims=3, grid_splits=6)
    gt = _true_neighbors(synthetic_corpus, synthetic_corpus[:25], 10, metric)
    prev = 0.0
    for n_probe in [1, 2, 4, 8]:
        r = _mean_recall(eng, synthetic_corpus, gt, 10, n_probe)
        assert r >= prev - 0.05, f"recall regressed at n_probe={n_probe}: {r:.2f} < {prev:.2f}"
        prev = r
    assert prev >= 0.15  # well above random (10/500)


def test_high_probe_recall_beats_random(synthetic_corpus):
    eng = MultiProbeSimilarity(synthetic_corpus, metric="cosine", grid_dims=3, grid_splits=4)
    gt = _true_neighbors(synthetic_corpus, synthetic_corpus[:25], 10, "cosine")
    assert _mean_recall(eng, synthetic_corpus, gt, 10, n_probe=8) >= 0.3


@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
def test_zero_vector_does_not_crash(synthetic_corpus, metric):
    eng = MultiProbeSimilarity(synthetic_corpus, metric=metric, grid_dims=3, grid_splits=6)
    res = eng.query_vector(np.zeros(synthetic_corpus.shape[1], dtype=np.float32), k=5, n_probe=4)
    assert len(res) > 0
    assert all(isinstance(r, (int, np.integer)) for r in res)


def test_far_out_of_bounds_query_uses_fallback(synthetic_corpus):
    eng = MultiProbeSimilarity(synthetic_corpus, metric="euclidean", grid_dims=3, grid_splits=6)
    huge = np.full(synthetic_corpus.shape[1], 1e6, dtype=np.float32)
    res = eng.query_vector(huge, k=3, n_probe=1)  # primary cell almost certainly empty
    assert len(res) > 0


def test_deterministic_across_builds():
    corpus = np.random.default_rng(5).random((200, 16)).astype(np.float32)
    q = corpus[3]
    a = MultiProbeSimilarity(corpus, metric="cosine", grid_dims=3, grid_splits=5)
    b = MultiProbeSimilarity(corpus, metric="cosine", grid_dims=3, grid_splits=5)
    assert a.query_vector(q, k=10, n_probe=4) == b.query_vector(q, k=10, n_probe=4)


def test_k_one_returns_single_result(synthetic_corpus):
    eng = MultiProbeSimilarity(synthetic_corpus, metric="cosine", grid_dims=3, grid_splits=6)
    res = eng.query_vector(synthetic_corpus[0], k=1, n_probe=4)
    assert len(res) == 1


@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
def test_single_cell_grid_is_exact(metric):
    # grid_splits=1 -> one cell -> every query gathers all points and re-ranks
    # exactly, so multiprobe must equal brute force.
    corpus = np.random.default_rng(8).standard_normal((40, 6)).astype(np.float32)
    q = np.random.default_rng(9).standard_normal(6).astype(np.float32)
    eng = MultiProbeSimilarity(corpus, metric=metric, grid_dims=2, grid_splits=1)
    bf = BruteForceSimilarity(corpus, metric=metric)
    assert eng.query_vector(q, k=5, n_probe=1) == bf.query_vector(q, k=5)
