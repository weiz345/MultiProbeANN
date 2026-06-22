from collections import deque

import numpy as np
from scipy.optimize import linear_sum_assignment


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------


def cosine_vector(embeddings, vec):
    # matmul trips spurious FP-flag RuntimeWarnings on macOS arm64 (Accelerate
    # BLAS); the results are correct, so ignore the flags.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        dots = embeddings @ vec
    denom = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(vec) + 1e-8
    return dots / denom


def cosine_pair(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-8)


def euclidean_vector(embeddings, vec):
    return -np.linalg.norm(embeddings - vec, axis=1)


def euclid_pair(vec_a, vec_b):
    return -np.linalg.norm(vec_a - vec_b)


# ------------------------------------------------------------
# Ranking / Hashing helpers
# ------------------------------------------------------------


def topk_from_scores(scores, k):
    if k == 1:
        return [int(np.argmax(scores))]
    return np.argsort(scores)[::-1][:k].tolist()


def bruteforce_query_vector(metric_vector_fn, vec, k):
    scores = metric_vector_fn(vec)
    return topk_from_scores(scores, k)


def hash_bits(vec, planes):
    signs = (planes @ vec) > 0
    out = 0
    for sign in signs:
        out = (out << 1) | int(sign)
    return out


# ------------------------------------------------------------
# Grid helpers
# ------------------------------------------------------------


class CellListND:
    """N-D grid; each cell -> list of point indices"""

    def __init__(self, dims, bounds, splits):
        self.dims = dims
        self.splits = splits
        self.mins = [bound[0] for bound in bounds]
        self.maxs = [bound[1] for bound in bounds]

        # Guard against zero-width dimensions (e.g. a constant/degenerate PCA
        # axis where max == min): a 0 cell_size would make _cell_index divide
        # by zero and produce NaN.  Fall back to 1.0 so every point maps to
        # cell 0 along that dimension.
        self.cell_size = [
            size if (size := (self.maxs[dim] - self.mins[dim]) / splits[dim]) > 0
            else 1.0
            for dim in range(dims)
        ]

        self.cells = {}

    def _cell_index(self, coord):
        idx = []
        for dim in range(self.dims):
            ratio = (coord[dim] - self.mins[dim]) / self.cell_size[dim]
            cell_index = int(ratio)
            cell_index = max(0, min(self.splits[dim] - 1, cell_index))
            idx.append(cell_index)
        return tuple(idx)

    def insert(self, coord, index_id):
        cell_id = self._cell_index(coord)
        if cell_id not in self.cells:
            self.cells[cell_id] = []
        self.cells[cell_id].append(index_id)

    def _lin(self, cell_id):
        result = 0
        multiplier = 1
        for dim in range(self.dims - 1, -1, -1):
            result += cell_id[dim] * multiplier
            multiplier *= self.splits[dim]
        return result

    def _unlin(self, value):
        result = []
        for dim in range(self.dims):
            multiplier = 1
            for dim2 in range(dim + 1, self.dims):
                multiplier *= self.splits[dim2]
            result.append((value // multiplier) % self.splits[dim])
        return tuple(result)

    def neighbors(self, value):
        cell_id = self._unlin(value)

        for dim in range(self.dims):
            plus_neighbor = list(cell_id)
            plus_neighbor[dim] += 1
            if 0 <= plus_neighbor[dim] < self.splits[dim]:
                yield self._lin(tuple(plus_neighbor))

            minus_neighbor = list(cell_id)
            minus_neighbor[dim] -= 1
            if 0 <= minus_neighbor[dim] < self.splits[dim]:
                yield self._lin(tuple(minus_neighbor))


# ------------------------------------------------------------
# BFS helpers
# ------------------------------------------------------------


def bfs_nearest_nonempty(total_nodes, neighbors_fn, occupied_nodes):
    """
    Generic multi-source BFS usable for:
        - Grid (3D neighbors)
    """
    INF = 10**9
    dist = [INF] * total_nodes
    nearest = [-1] * total_nodes

    queue = deque()

    for node in occupied_nodes:
        dist[node] = 0
        nearest[node] = node
        queue.append(node)

    while queue:
        node = queue.popleft()
        for neighbor in neighbors_fn(node):
            if dist[neighbor] == INF:
                dist[neighbor] = dist[node] + 1
                nearest[neighbor] = nearest[node]
                queue.append(neighbor)

    return dist, nearest


def hamming_neighbors(code, num_bits):
    """Generate neighbors by flipping exactly 1 bit."""
    for bit in range(num_bits):
        yield code ^ (1 << bit)
