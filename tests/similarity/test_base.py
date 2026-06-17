import numpy as np
import pytest

from ann.similarity.base import SimilarityBase
from ann.similarity.backends.bruteforce import BruteForceSimilarity


def test_metric_dispatch_cosine_vs_euclidean():
    x = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]], dtype=np.float32)
    cos = SimilarityBase(x, metric="cosine")
    euc = SimilarityBase(x, metric="euclidean")

    q = np.array([1.0, 0.0], dtype=np.float32)
    # cosine: [1,0] and [2,0] are colinear with q -> score 1.0; [0,1] -> 0.0
    cos_scores = cos.metric_vector_fn(q)
    assert cos_scores[0] == pytest.approx(1.0, abs=1e-4)
    assert cos_scores[2] == pytest.approx(1.0, abs=1e-4)
    assert cos_scores[1] == pytest.approx(0.0, abs=1e-4)
    # euclidean returns negative distance: nearest (row 0, dist 0) is largest
    euc_scores = euc.metric_vector_fn(q)
    assert np.argmax(euc_scores) == 0


def test_base_query_fn_not_implemented():
    x = np.random.default_rng(0).random((10, 4)).astype(np.float32)
    base = SimilarityBase(x, metric="cosine")
    with pytest.raises(NotImplementedError):
        base.query_fn(0, 3)
    with pytest.raises(NotImplementedError):
        base.query_vector_fn(x[0], 3)


@pytest.mark.parametrize(
    "k, exc, match",
    [
        (0, ValueError, "positive integer"),
        (-1, ValueError, "positive integer"),
        (1.5, TypeError, "integer"),
        (True, TypeError, "integer"),
    ],
)
def test_k_validation(synthetic_corpus, k, exc, match):
    sim = BruteForceSimilarity(synthetic_corpus, metric="cosine")
    with pytest.raises(exc, match=match):
        sim.query(0, k=k)
    with pytest.raises(exc, match=match):
        sim.query_vector(synthetic_corpus[0], k=k)


def test_query_excludes_self(synthetic_corpus):
    sim = BruteForceSimilarity(synthetic_corpus, metric="cosine")
    for idx in (0, 5, 123):
        results = sim.query(idx, k=10)
        assert idx not in results
        assert len(results) == 10


def test_query_overflow_returns_all():
    x = np.random.default_rng(1).standard_normal((20, 5)).astype(np.float32)
    sim = BruteForceSimilarity(x, metric="cosine")
    # query(idx) excludes self -> at most N-1 results
    assert len(sim.query(0, k=100)) == 19
    # query_vector keeps all candidates
    assert len(sim.query_vector(x[0], k=100)) == 20
