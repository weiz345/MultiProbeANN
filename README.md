# Multi-Probe Grid

An implementation of **Multi-Probe Grid** for approximate
nearest neighbor (ANN) search, packaged for use with
[ann-benchmarks](https://github.com/erikbern/ann-benchmarks).

The algorithm projects vectors into a low-dimensional space with PCA, buckets them
into an N-dimensional grid, and answers queries by probing the query's primary cell
plus its nearest neighboring cells (`n_probe`). Cells are ranked by wall distance
from the query vector.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Or install directly from the source tree's requirements:

```bash
pip install -r requirements.txt
```

The package depends only on `numpy`, `scikit-learn`, and `scipy`.

## Usage

```python
import numpy as np
from ann.similarity.backends.multiprobe import MultiProbeSimilarity

corpus = np.random.rand(10000, 128).astype(np.float32)

engine = MultiProbeSimilarity(
    corpus,
    metric="euclidean",   # "euclidean" or "cosine"
    grid_dims=3,          # PCA dimensionality
    grid_splits=6,        # grid granularity per dimension
)

query = np.random.rand(128).astype(np.float32)
neighbors = engine.query_vector(query, k=10, n_probe=4)
```

### Hyperparameters

| Parameter | Type | When | Description |
|---|---|---|---|
| `grid_dims` | int | build-time | Number of PCA dimensions to project into |
| `grid_splits` | int or tuple | build-time | Number of grid cells per dimension |
| `n_probe` | int | query-time | Number of cells probed (1 = plain grid) |

## Testing

```bash
python -m pytest -q
```

## ann-benchmarks Integration

This package can be plugged into [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) as a standalone algorithm. The integration files live in `ann_benchmarks_integration/`.

### Setup

Run the setup script, pointing it at your local ann-benchmarks clone:

```bash
./ann_benchmarks_integration/setup.sh <path/to/ann-benchmarks>

# Example:
# ./ann_benchmarks_integration/setup.sh ../ann-benchmarks
```

This copies the integration files (`module.py`, `config.yml`, `Dockerfile`) into
`ann_benchmarks/algorithms/ann_multiprobe/` and builds the Docker image.

On **ARM64 / Apple Silicon**, the script automatically detects the architecture and
patches the upstream `hnswlib`, `faiss`, and `faiss_hnsw` Dockerfiles so they build
natively (the originals download x86-only Anaconda binaries). The original files are
saved as `Dockerfile.x86_backup`; on x86_64 systems no patching is done.

<details>
<summary>Manual setup (without the script)</summary>

1. **Fork ann-benchmarks** and copy the integration files:

```bash
# Inside your ann-benchmarks fork:
mkdir -p ann_benchmarks/algorithms/ann_multiprobe
cp /path/to/ann_benchmarks_integration/module.py   ann_benchmarks/algorithms/ann_multiprobe/
cp /path/to/ann_benchmarks_integration/config.yml  ann_benchmarks/algorithms/ann_multiprobe/
cp /path/to/ann_benchmarks_integration/Dockerfile  ann_benchmarks/algorithms/ann_multiprobe/
```

2. **Build the Docker image**:

```bash
cd /path/to/ann-benchmarks
python install.py --algorithm ann_multiprobe
```

</details>

### Running benchmarks

All commands below run from inside the ann-benchmarks directory with its dependencies available:

```bash
cd <path/to/ann-benchmarks>
source .venv/bin/activate    # or whichever env has ann-benchmarks installed
```

Build the Docker image first:

```bash
python install.py --algorithm ann_multiprobe
```

`install.py` uses the directory name (underscore: `ann_multiprobe`), while `run.py` uses the algorithm name from config.yml (hyphen: `ann-multiprobe`).

Run a single algorithm:

```bash
python run.py --algorithm ann-multiprobe --dataset fashion-mnist-784-euclidean
```

Run all algorithms for a dataset (omit `--algorithm`). Only algorithms with built Docker images will run:

```bash
python run.py --dataset fashion-mnist-784-euclidean
```

Run multiple specific algorithms separately — results accumulate in `results/` and are combined automatically when plotting:

```bash
python install.py --algorithm ann_multiprobe
python install.py --algorithm annoy
python install.py --algorithm hnswlib

python run.py --dataset fashion-mnist-784-euclidean --algorithm ann-multiprobe
python run.py --dataset fashion-mnist-784-euclidean --algorithm annoy
python run.py --dataset fashion-mnist-784-euclidean --algorithm hnswlib
```

Use `--max-n-algorithms N` to limit the number of algorithms in a full run, or `--parallelism N` to run multiple Docker containers in parallel.

### Plotting results

```bash
python plot.py --dataset fashion-mnist-784-euclidean
```

The reference images on the ann-benchmarks website use a **log scale** for the Y-axis (QPS). To match that style, pass `-Y log`:

```bash
python plot.py --dataset fashion-mnist-784-euclidean -Y log
```

### Parameter mapping

| ann-benchmarks | This package | When |
|---|---|---|
| `args: [[2,3,4], [6,8,10]]` | `grid_dims`, `grid_splits` | Build-time (one index per combo) |
| `query_args: [[1,2,4,8,16]]` | `n_probe` | Query-time (reuses the same index) |

The config sweeps `grid_dims × grid_splits` as build-time (index) arguments and `n_probe` as a query-time argument, so each index is queried with multiple probe counts without rebuilding.

## Project Structure

```
.
├── src/
│   └── ann/
│       └── similarity/
│           ├── base.py             # SimilarityBase + metric helpers
│           ├── utils.py            # Grid cell list, BFS, metric functions
│           └── backends/
│               └── multiprobe.py   # Multi-Probe Grid algorithm
├── ann_benchmarks_integration/    # Files for the ann-benchmarks framework
│   ├── module.py                   # BaseANN wrapper
│   ├── config.yml                  # Parameter sweep definition
│   ├── Dockerfile                  # Docker build for ann-benchmarks
│   └── setup.sh                    # Copies files into an ann-benchmarks clone
├── tests/                         # Integration test
├── pyproject.toml                 # Package metadata (pip install -e .)
└── requirements.txt               # Dependencies
```

## License

MIT

