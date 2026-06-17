import numpy as np
import pytest

from ann.similarity.utils import (
    CellListND,
    bfs_nearest_nonempty,
    bruteforce_query_vector,
    cosine_pair,
    cosine_vector,
    euclid_pair,
    euclidean_vector,
    hamming_neighbors,
    topk_from_scores,
)


def test_cosine_vector_colinear_and_orthogonal():
    emb = np.array([[1.0, 0.0], [0.0, 1.0], [3.0, 0.0]], dtype=np.float32)
    scores = cosine_vector(emb, np.array([1.0, 0.0], dtype=np.float32))
    assert scores[0] == pytest.approx(1.0, abs=1e-4)
    assert scores[2] == pytest.approx(1.0, abs=1e-4)
    assert scores[1] == pytest.approx(0.0, abs=1e-4)


def test_euclidean_vector_is_negative_distance():
    emb = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    scores = euclidean_vector(emb, np.array([0.0, 0.0], dtype=np.float32))
    assert scores[0] == pytest.approx(0.0, abs=1e-5)
    assert scores[1] == pytest.approx(-5.0, abs=1e-4)


def test_metric_pairs():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert cosine_pair(a, a) == pytest.approx(1.0, abs=1e-4)
    assert cosine_pair(a, b) == pytest.approx(0.0, abs=1e-4)
    assert euclid_pair(a, a) == pytest.approx(0.0, abs=1e-5)
    assert euclid_pair(a, b) == pytest.approx(-np.sqrt(2), abs=1e-4)


def test_topk_from_scores_k1_and_kn():
    scores = np.array([0.1, 0.9, 0.5, 0.2], dtype=np.float32)
    assert topk_from_scores(scores, 1) == [1]
    assert topk_from_scores(scores, 3) == [1, 2, 3]


def test_bruteforce_query_vector_matches_argsort():
    emb = np.random.default_rng(3).standard_normal((30, 6)).astype(np.float32)

    def metric_fn(v):
        return emb @ v

    q = emb[4]
    got = bruteforce_query_vector(metric_fn, q, 5)
    expected = np.argsort(emb @ q)[::-1][:5].tolist()
    assert got == expected


def test_cell_index_basic_and_clamps_out_of_range():
    grid = CellListND(2, [(0.0, 1.0), (0.0, 1.0)], (3, 3))
    assert grid._cell_index((0.5, 0.5)) == (1, 1)
    # Far outside the bounds snaps to the edge cells, no IndexError.
    assert grid._cell_index((-99.0, -99.0)) == (0, 0)
    assert grid._cell_index((99.0, 99.0)) == (2, 2)


def test_cell_index_zero_width_dimension_guard():
    # A degenerate (constant) dimension must not divide by zero / produce NaN.
    grid = CellListND(2, [(0.0, 0.0), (0.0, 1.0)], (3, 3))
    cid = grid._cell_index((0.0, 0.5))
    assert cid[0] == 0
    assert all(isinstance(c, int) for c in cid)


def test_lin_unlin_round_trip():
    grid = CellListND(2, [(0.0, 1.0), (0.0, 1.0)], (3, 3))
    for a in range(3):
        for b in range(3):
            lin = grid._lin((a, b))
            assert grid._unlin(lin) == (a, b)


def test_neighbors_stay_in_bounds():
    grid = CellListND(2, [(0.0, 1.0), (0.0, 1.0)], (3, 3))
    corner = grid._lin((0, 0))
    neigh = {grid._unlin(n) for n in grid.neighbors(corner)}
    assert neigh == {(0, 1), (1, 0)}  # no negative cells


def test_bfs_maps_empty_cells_to_nearest_occupied():
    grid = CellListND(2, [(0.0, 1.0), (0.0, 1.0)], (3, 3))
    grid.insert((0.1, 0.1), 0)  # occupies cell (0,0) -> lin 0
    occupied = [grid._lin(cid) for cid in grid.cells]
    dist, nearest = bfs_nearest_nonempty(9, grid.neighbors, occupied)

    far = grid._lin((2, 2))
    assert nearest[far] == 0
    assert dist[far] == 4  # Manhattan distance across the grid


def test_hamming_neighbors_flips_one_bit():
    assert set(hamming_neighbors(0b000, 3)) == {0b001, 0b010, 0b100}
