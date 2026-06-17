import h5py
import numpy as np
import pytest


@pytest.fixture
def synthetic_corpus():
    """Seeded (500, 64) float32 corpus."""
    return np.random.default_rng(42).random((500, 64)).astype(np.float32)


@pytest.fixture
def unit_corpus():
    """Row-normalized (60, 16) corpus for cosine/euclidean rank equivalence."""
    x = np.random.default_rng(123).standard_normal((60, 16)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x


def brute_force_neighbors(train, test, k, metric):
    """Ground-truth (Q, k) neighbor indices via exhaustive search."""
    out = []
    for q in test:
        if metric == "angular" or metric == "cosine":
            denom = np.linalg.norm(train, axis=1) * np.linalg.norm(q) + 1e-9
            scores = (train @ q) / denom
            out.append(np.argsort(scores)[::-1][:k])
        else:
            d = np.linalg.norm(train - q, axis=1)
            out.append(np.argsort(d)[:k])
    return np.array(out, dtype=np.int32)


def neighbors_with_recall(gt, recall, k=10):
    """Result neighbors matching gt in the first round(recall*k) columns.

    Remaining slots are filled with -1 (never intersects the non-negative gt
    indices), so per-query recall@k is exactly round(recall*k)/k.
    """
    m = round(recall * k)
    res = np.full((gt.shape[0], k), -1, dtype=np.int32)
    res[:, :m] = gt[:, :m]
    return res


@pytest.fixture
def tiny_annbench_hdf5(tmp_path):
    """Minimal ann-benchmarks dataset HDF5 (train/test/neighbors); returns path."""
    rng = np.random.default_rng(7)
    train = rng.random((200, 16)).astype(np.float32)
    test = rng.random((20, 16)).astype(np.float32)
    neighbors = brute_force_neighbors(train, test, 100, "angular")
    path = tmp_path / "tiny.hdf5"
    with h5py.File(path, "w") as f:
        f.create_dataset("train", data=train)
        f.create_dataset("test", data=test)
        f.create_dataset("neighbors", data=neighbors)
    return str(path)


@pytest.fixture
def add_dataset(tmp_path):
    """Factory building an ann-benchmarks-style results tree under one ann_dir.

    Call as add_dataset(name, gt, specs) where gt is (Q, k) ground-truth and
    specs is a list of dicts {algo, name, best_time, neighbors}. Returns the
    shared ann_dir path (str) so multiple datasets accumulate under it.
    """
    ann_dir = tmp_path / "annbench"

    def _add(dataset, gt, specs):
        data_dir = ann_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(data_dir / f"{dataset}.hdf5", "w") as f:
            f.create_dataset("neighbors", data=gt)
        for s in specs:
            d = ann_dir / "results" / dataset / "10" / s["algo"]
            d.mkdir(parents=True, exist_ok=True)
            with h5py.File(d / f"{s['name']}.hdf5", "w") as f:
                f.attrs["best_search_time"] = float(s["best_time"])
                f.create_dataset("neighbors", data=s["neighbors"])
        return str(ann_dir)

    return _add
