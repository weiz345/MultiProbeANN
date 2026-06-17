import numpy as np
import pytest

from ann.similarity.backends.bruteforce import BruteForceSimilarity


@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
def test_query_vector_is_exact(metric):
    corpus = np.random.default_rng(2).standard_normal((60, 8)).astype(np.float32)
    q = np.random.default_rng(3).standard_normal(8).astype(np.float32)
    bf = BruteForceSimilarity(corpus, metric=metric)
    if metric == "cosine":
        denom = np.linalg.norm(corpus, axis=1) * np.linalg.norm(q) + 1e-9
        expected = np.argsort((corpus @ q) / denom)[::-1][:5].tolist()
    else:
        expected = np.argsort(np.linalg.norm(corpus - q, axis=1))[:5].tolist()
    assert bf.query_vector(q, k=5) == expected


def test_query_excludes_self(synthetic_corpus):
    bf = BruteForceSimilarity(synthetic_corpus, metric="cosine")
    assert 7 not in bf.query(7, k=10)


def test_overflow_returns_all():
    corpus = np.random.default_rng(4).standard_normal((15, 5)).astype(np.float32)
    bf = BruteForceSimilarity(corpus, metric="euclidean")
    assert len(bf.query_vector(corpus[0], k=999)) == 15


def test_cosine_euclidean_rank_equivalence_on_unit_vectors(unit_corpus):
    # On unit vectors the two metrics induce the same ranking.
    bf_cos = BruteForceSimilarity(unit_corpus, metric="cosine")
    bf_euc = BruteForceSimilarity(unit_corpus, metric="euclidean")
    q = unit_corpus[0]
    cos_rank = np.argsort(bf_cos.metric_vector_fn(q))[::-1]
    euc_rank = np.argsort(bf_euc.metric_vector_fn(q))[::-1]
    assert np.array_equal(cos_rank, euc_rank)


def test_results_are_ints(synthetic_corpus):
    bf = BruteForceSimilarity(synthetic_corpus, metric="cosine")
    res = bf.query_vector(synthetic_corpus[0], k=5)
    assert all(isinstance(r, (int, np.integer)) for r in res)
