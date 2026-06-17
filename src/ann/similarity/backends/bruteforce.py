import numpy as np

from ..base import SimilarityBase
from ..utils import bruteforce_query_vector, topk_from_scores


class BruteForceSimilarity(SimilarityBase):
    def __init__(self, embeddings, metric="cosine"):
        super().__init__(embeddings, metric=metric, backend="bruteforce")

        self._profile_setup(lambda: None)

    def query_fn(self, idx, k):
        vec = self.embeddings[idx]
        sims = self.metric_vector_fn(vec)
        sims[idx] = -np.inf

        return topk_from_scores(sims, k)

    def query_vector_fn(self, vec, k):
        return bruteforce_query_vector(self.metric_vector_fn, vec, k)
