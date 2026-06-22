# Multi-Probe Grid

An implementation of **Multi-Probe Grid** for approximate nearest neighbor (ANN)
search, packaged for use with
[ann-benchmarks](https://github.com/erikbern/ann-benchmarks). This is the
reference code for the paper *Scaling Laws for Grid-Based Approximate Nearest
Neighbor Search in High Dimensions* (HiLD workshop).

The algorithm projects vectors into a low-dimensional space with PCA, buckets them
into an N-dimensional grid, and answers queries by probing the query's primary cell
plus its nearest neighboring cells (`n_probe`), ranked by wall distance, then
re-ranking candidates in the native space.

## Get started

New here? Follow **[docs/TUTORIAL.md](docs/TUTORIAL.md)** to go from a clean
machine to a real QPS-vs-recall Pareto front in about ten minutes. It runs the
same workflow as the paper's Fig 1 on a small, fast dataset.

## Install

```bash
conda env create -f environment.yml
conda activate multiprobe-ann
```

This installs the package editable with the `repro` + `dev` extras (figure and
tuning dependencies plus pytest). For just the core algorithm:

```bash
pip install -e .
```

The core package depends only on `numpy`, `scikit-learn`, and `scipy`.

This `environment.yml` env is enough to use the algorithm, run the tests, and
re-plot from existing results. Running the Docker benchmark sweep yourself needs
ann-benchmarks' own dependencies as well; [docs/TUTORIAL.md](docs/TUTORIAL.md)
gives a one-environment recipe that covers both.

## Quickstart

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

## ann-benchmarks integration

The integration files live in `ann_benchmarks_integration/`. Point the setup
script at your local ann-benchmarks clone:

```bash
./ann_benchmarks_integration/setup.sh <path/to/ann-benchmarks>
```

This copies `module.py`, `config.yml`, and `Dockerfile` into
`ann_benchmarks/algorithms/ann_multiprobe/` and builds the Docker image. On
ARM64 / Apple Silicon it also patches the upstream `hnswlib`, `faiss`, and
`faiss_hnsw` Dockerfiles so they build natively (originals saved as
`Dockerfile.x86_backup`).

Then, from inside the ann-benchmarks clone:

```bash
python run.py --algorithm ann-multiprobe --dataset glove-100-angular
python plot.py --dataset glove-100-angular -Y log
```

The shipped `config.yml` defines a single representative operating point
(`grid_dims=6, grid_splits=5` swept over `n_probe`). The full per-dataset Pareto
sweeps used in the paper are regenerated with `tuning/` -- see
[docs/REPRODUCTION.md](docs/REPRODUCTION.md).

## Reproducing the paper figures

[docs/REPRODUCTION.md](docs/REPRODUCTION.md) walks through reproducing **Fig 1**
(Pareto fronts on glove-200-angular) and **Fig 2b** (the d-scaling crossover):
derive the multiprobe configs with `tuning/optuna_search.py`, run the benchmark
sweep with `reproduce/run_benchmarks.sh`, then render the figures with
`reproduce/fig1_pareto.py` and `reproduce/fig2b_dscaling.py`.

## Project structure

```
.
├── src/ann/similarity/           # the algorithm package
│   ├── base.py  utils.py         # SimilarityBase, grid/BFS/metric helpers
│   └── backends/
│       ├── multiprobe.py         # Multi-Probe Grid
│       └── bruteforce.py         # exact reference baseline
├── ann_benchmarks_integration/   # files copied into an ann-benchmarks clone
├── tuning/                       # NSGA-II search + Pareto-config extraction
├── reproduce/                    # Fig 1 + Fig 2b scripts and shared helpers
├── configs/                      # tuning search configs
├── tests/                        # mirrored pytest suite
└── docs/REPRODUCTION.md          # full reproduction walkthrough
```

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{multiprobegrid2026,
  title     = {Scaling Laws for Grid-Based Approximate Nearest Neighbor Search in High Dimensions},
  author    = {TODO},
  booktitle = {HiLD Workshop},
  year      = {TODO},
  note      = {TODO: eprint / URL},
}
```

## Contact

Questions and bug reports: please open an issue on
[GitHub](https://github.com/weiz345/MultiProbeANN/issues).

## Contributing

Contributions are welcome. Open an issue to discuss a change, or send a pull
request. Please run the test suite before submitting:

```bash
python -m pytest -q
```

## License

MIT
