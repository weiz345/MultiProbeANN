"""
Extends the grid approach with multiprobe search. 
if n_probe = 1, we do default grid
if n_probe = 2, we check the primary cell + nearest neighboring cell 
if n_probe = 3, we check the primary cell + 2 nearest neighboring cells

etc. 

nearest cells are ranked by wall distance from query vector 

Hyperparameters:
- PCA dimensionality  (grid_dims)
- Grid granularity    (grid_splits)
- Multi-probe breadth (n_probe, passed at query time)
"""

import itertools
from typing import List, Optional, Tuple, Union
import numpy as np
from sklearn.decomposition import PCA
from ..base import SimilarityBase
from ..utils import CellListND, bfs_nearest_nonempty


class MultiProbeSimilarity(SimilarityBase):
    """
    Multi-Probe Grid Search backend.

    Parameters
    ----------
    embeddings : np.ndarray
        Corpus vectors, shape (N, D).
    metric : str
        "cosine" or "euclidean".
    grid_dims : int
        Number of PCA dimensions to project into.
    grid_splits : int or tuple of int
        Number of grid cells per dimension.  If an int, the same value is
        used for every dimension.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        metric: str = "cosine",
        grid_dims: int = 3,
        grid_splits: Union[int, Tuple[int, ...]] = 6,
    ) -> None:
        super().__init__(embeddings, metric=metric, backend="multiprobe")

        def setup() -> None:
            splits: Tuple[int, ...] = (
                tuple([grid_splits] * grid_dims)
                if isinstance(grid_splits, int)
                else grid_splits
            )
            self._init_multiprobe_backend(grid_dims, splits)

        self._profile_setup(setup)

    # Index construction
    def _init_multiprobe_backend(self, dims: int, splits: Tuple[int, ...]) -> None:
        pca = PCA(n_components=dims)
        projected = pca.fit_transform(self.embeddings)

        # Store raw numpy arrays for fast manual projection at query time
        # (avoiding sklearn transform overhead per query)
        self.pca_comp: np.ndarray = pca.components_.T.astype(np.float32)  # (D, dims)
        self.pca_mean: np.ndarray = pca.mean_.astype(np.float32)           # (D,)

        # Pre-compute L2 norms for cosine similarity (reused across all queries)
        self.embedding_norms: Optional[np.ndarray]
        if self.metric == "cosine":
            self.embedding_norms = np.linalg.norm(
                self.embeddings, axis=1
            ).astype(np.float32)
        else:
            self.embedding_norms = None

        # build grid
        bounds: List[Tuple[float, float]] = [
            (float(projected[:, d].min()), float(projected[:, d].max()))
            for d in range(dims)
        ]
        self.grid: CellListND = CellListND(dims, bounds, splits)

        for idx, point in enumerate(projected):
            self.grid.insert(tuple(point), idx)

        # BFS
        total: int = 1
        for s in splits:
            total *= s
        occupied: List[int] = [self.grid._lin(cell_id) for cell_id in self.grid.cells]

        self.grid_dist, self.grid_nearest = bfs_nearest_nonempty(
            total_nodes=total,
            neighbors_fn=self.grid.neighbors,
            occupied_nodes=occupied,
        )

    # SimilarityBase interface
    def query_fn(self, idx: int, k: int) -> List[int]:
        return self.query_vector_fn(self.embeddings[idx], k)

    def query_vector_fn(self, vec: np.ndarray, k: int) -> List[int]:
        """Called by SimilarityBase.query_vector — uses default n_probe=4."""
        return self._grid_query_vector_multiprobe(vec, k, n_probe=4)

    def query_vector(self, vec: np.ndarray, k: int = 5, n_probe: int = 4) -> List[int]:
        """
        Override to expose n_probe at the call site.

        Parameters
        ----------
        vec : np.ndarray
            Query vector.
        k : int
            Number of nearest neighbours to return.
        n_probe : int
            Number of grid cells to probe (1 = standard single-cell grid search).
        """
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
            raise TypeError("k must be an integer")
        if k <= 0:
            raise ValueError("k must be a positive integer")
        return self._grid_query_vector_multiprobe(vec, k, n_probe)

    # core logic 
    def _grid_query_vector_multiprobe(
        self, vec: np.ndarray, k: int, n_probe: int
    ) -> List[int]:
        """
        Multi-Probe implementation for the Grid backend.

        Steps
        -----
        1. Project query vector via stored PCA weights (no sklearn overhead).
        2. Identify the primary cell.
        3. Find the n_probe nearest cells (ranked by wall distance).
        4. Gather candidates from those cells.
        5. If all probed cells are empty, fall back to BFS nearest non-empty cell.
        6. Re-rank candidates with the full metric and return top-k indices.
        """
        # 1. Fast manual PCA projection
        pt: np.ndarray = (vec - self.pca_mean) @ self.pca_comp

        # 2. Primary cell
        primary_cid: Tuple[int, ...] = self.grid._cell_index(pt)

        # 3. Top-n_probe nearest cells (includes primary)
        probed_cells: List[Tuple[int, ...]] = self._get_nearest_cells(pt, primary_cid, n_probe)

        # 4. Gather candidates
        candidate_lists: List[List[int]] = [
            self.grid.cells[cid] for cid in probed_cells if cid in self.grid.cells
        ]

        # 5. Fallback: BFS to nearest non-empty cell
        if not candidate_lists:
            u: int = self.grid._lin(primary_cid)
            nearest_u: int = self.grid_nearest[u]
            nearest_cid: Tuple[int, ...] = self.grid._unlin(nearest_u)
            if nearest_cid in self.grid.cells:
                candidate_lists = [self.grid.cells[nearest_cid]]

        if not candidate_lists:
            return []

        candidate_indices: np.ndarray = np.concatenate(candidate_lists).astype(np.int32)
        candidate_vectors: np.ndarray = self.embeddings[candidate_indices]

        # 6. Re-rank with full metric
        top_k_local: np.ndarray
        if self.metric == "cosine":
            d_q: float = float(np.linalg.norm(vec))
            d_c: np.ndarray = (
                self.embedding_norms[candidate_indices]
                if self.embedding_norms is not None
                else np.linalg.norm(candidate_vectors, axis=1)
            )
            scores: np.ndarray = (candidate_vectors @ vec) / (d_q * d_c + 1e-9)

            if len(scores) <= k:
                top_k_local = np.argsort(scores)[::-1]
            else:
                idx = np.argpartition(scores, -k)[-k:]
                top_k_local = idx[np.argsort(scores[idx])[::-1]]

        else:  # euclidean — smallest distance wins
            dists: np.ndarray = np.linalg.norm(candidate_vectors - vec, axis=1)

            if len(dists) <= k:
                top_k_local = np.argsort(dists)
            else:
                idx = np.argpartition(dists, k)[:k]
                top_k_local = idx[np.argsort(dists[idx])]

        results: List[int] = candidate_indices[top_k_local].tolist()

        if k == 1 and results:
            return results[:1]

        return results

    def _get_nearest_cells(
        self,
        pt: np.ndarray,
        center_cid: Tuple[int, ...],
        n_probe: int,
    ) -> List[Tuple[int, ...]]:
        """
        Return the n_probe cells closest to query point *pt*, starting with
        the primary cell.

        For the sake of speed, only examines the 2^D quadrant/octant neighbours the
        point is geometrically closest to, rather than all 3^D neighbours.
        All distance calculations are vectorised with numpy.

        Parameters
        ----------
        pt : np.ndarray
            Query point in PCA space.
        center_cid : tuple of int
            Cell index tuple of the primary cell.
        n_probe : int
            Total number of cells to return (including the primary).

        Returns
        -------
        list of tuple of int
            Cell index tuples, primary cell first, then nearest neighbours.
        """
        dims: int = self.grid.dims

        # Fast path: single-probe, no neighbour search needed
        if n_probe <= 1:
            return [center_cid]

        center_arr: np.ndarray = np.array(center_cid)
        pt_arr: np.ndarray = np.array(pt)
        grid_splits: np.ndarray = np.array(self.grid.splits)
        cell_size: np.ndarray = np.array(self.grid.cell_size)
        grid_mins: np.ndarray = np.array(self.grid.mins)

        # Cell boundaries for the primary cell
        cell_origins: np.ndarray = grid_mins + center_arr * cell_size
        cell_midpoints: np.ndarray = cell_origins + cell_size * 0.5

        # For each dimension, look only toward the side the point is closer to
        search_directions: List[Tuple[int, int]] = []
        for d in range(dims):
            if pt_arr[d] >= cell_midpoints[d]:
                search_directions.append((0, 1))   # right neighbour
            else:
                search_directions.append((-1, 0))  # left neighbour

        # Generate all 2^D offset combinations
        offsets: np.ndarray = np.array(
            list(itertools.product(*search_directions)), dtype=np.int32
        )  # shape (2^D, dims)

        neighbor_coords: np.ndarray = center_arr + offsets  # broadcasting (2^D, dims)

        # Filter: in-bounds and not the center cell itself
        in_bounds: np.ndarray = np.all(
            (neighbor_coords >= 0) & (neighbor_coords < grid_splits), axis=1
        )
        is_center: np.ndarray = np.all(offsets == 0, axis=1)
        valid_mask: np.ndarray = in_bounds & ~is_center

        valid_offsets: np.ndarray = offsets[valid_mask]
        valid_neighbors: np.ndarray = neighbor_coords[valid_mask]

        num_valid: int = int(valid_neighbors.shape[0])
        if num_valid == 0:
            return [center_cid]

        # Vectorised squared distance from pt to each shared cell wall
        cell_mins_arr: np.ndarray = cell_origins
        cell_maxs_arr: np.ndarray = cell_origins + cell_size
        dist_sq: np.ndarray = np.zeros(num_valid, dtype=np.float32)

        for d in range(dims):
            moves_right: np.ndarray = valid_offsets[:, d] > 0
            moves_left: np.ndarray = valid_offsets[:, d] < 0

            # Distance to right wall (for rightward neighbours)
            dr: float = float(pt_arr[d] - cell_maxs_arr[d])
            dist_sq[moves_right] += dr * dr

            # Distance to left wall (for leftward neighbours)
            dl: float = float(cell_mins_arr[d] - pt_arr[d])
            dist_sq[moves_left] += dl * dl

        # Pick the closest (n_probe - 1) neighbours; primary cell fills slot 0
        sorted_idx: np.ndarray = np.argsort(dist_sq)[: n_probe - 1]
        top_neighbors: List[Tuple[int, ...]] = [
            tuple(valid_neighbors[i]) for i in sorted_idx
        ]

        return [center_cid] + top_neighbors
