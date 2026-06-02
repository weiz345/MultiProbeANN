"""
ann-benchmarks wrapper for Multi-Probe Grid.

Drop this file into:
    ann_benchmarks/algorithms/ann_multiprobe/module.py
"""

import numpy as np

from ann.similarity.backends.multiprobe import MultiProbeSimilarity

try:
    from ..base.module import BaseANN
except (ImportError, SystemError, ValueError):

    class BaseANN:
        """Fallback stub when running outside the ann-benchmarks package."""

        def __init__(self):
            self.name = ""

        def __str__(self):
            return self.name


class AnnMultiprobe(BaseANN):
    """Multi-Probe Grid via PCA projection + grid bucketing."""

    def __init__(self, metric, grid_dims, grid_splits):
        self._metric = {"angular": "cosine", "euclidean": "euclidean"}[metric]
        self._grid_dims = int(grid_dims)
        self._grid_splits = int(grid_splits)
        self._n_probe = 1
        self.name = (
            f"AnnMultiprobe(dims={self._grid_dims}, "
            f"splits={self._grid_splits}, n_probe={self._n_probe})"
        )

    def fit(self, X):
        self._engine = MultiProbeSimilarity(
            X.astype(np.float32),
            metric=self._metric,
            grid_dims=self._grid_dims,
            grid_splits=self._grid_splits,
        )

    def set_query_arguments(self, n_probe):
        self._n_probe = int(n_probe)
        self.name = (
            f"AnnMultiprobe(dims={self._grid_dims}, "
            f"splits={self._grid_splits}, n_probe={self._n_probe})"
        )

    def query(self, q, n):
        return self._engine.query_vector(q, k=n, n_probe=self._n_probe)

    def __str__(self):
        return self.name
