import time
import tracemalloc

import numpy as np

from .utils import (
    cosine_pair,
    cosine_vector,
    euclid_pair,
    euclidean_vector,
)


class SimilarityBase:
    def __init__(self, embeddings, metric="cosine", backend="base"):
        self.embeddings = embeddings.astype(np.float32)
        self.metric = metric
        self.backend = backend

        if metric == "cosine":
            self.metric_vector_fn = lambda vec: cosine_vector(self.embeddings, vec)
            self.metric_pair = cosine_pair
        else:
            self.metric_vector_fn = lambda vec: euclidean_vector(self.embeddings, vec)
            self.metric_pair = euclid_pair

    def _profile_setup(self, setup_fn):
        tracemalloc.start()
        t0 = time.perf_counter()

        setup_fn()

        t1 = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.mem_peak_mb = peak / 1024 / 1024
        print(f"[{self.backend}] Init time: {(t1 - t0)*1000:.3f} ms")
        print(f"[{self.backend}] Mem peak: {peak/1024/1024:.4f} MB")

    def query(self, idx, k=5):
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
            raise TypeError("k must be an integer")
        if k <= 0:
            raise ValueError("k must be a positive integer")

        results = self.query_fn(idx, k + 1)
        if idx in results:
            results.remove(idx)
        return results[:k]

    def query_vector(self, vec, k=5):
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
            raise TypeError("k must be an integer")
        if k <= 0:
            raise ValueError("k must be a positive integer")
        return self.query_vector_fn(vec, k)

    def query_fn(self, idx, k):
        raise NotImplementedError("Backend must implement query_fn")

    def query_vector_fn(self, vec, k):
        raise NotImplementedError("Backend must implement query_vector_fn")
