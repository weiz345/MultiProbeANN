# Reproducing the paper figures

This walks through reproducing the two key figures of *Scaling Laws for
Grid-Based Approximate Nearest Neighbor Search in High Dimensions*:

- **Fig 1** -- Pareto fronts (QPS vs recall@10) on glove-200-angular.
- **Fig 2b** -- the d-scaling crossover: d-scaling exponent vs recall across
  glove-25/50/100/200-angular.

Both use only base datasets that ann-benchmarks auto-downloads; no subsampling is
required. (The N-scaling panel, Fig 2a, is not included here.)

For a fast first run on a small dataset, start with
[TUTORIAL.md](TUTORIAL.md); this document reproduces the full paper figures and is
the long path.

Data flow:

```
tuning/optuna_search.py ──▶ Optuna study ──▶ extract_pareto_configs.py ──▶ config.yml
                                                                               │
ann-benchmarks run.py (Docker, per algorithm) ◀───────────────────────────────┘
        │
        ▼
results/<dataset>/10/<algo>/*.hdf5 ──▶ reproduce/fig1_pareto.py / fig2b_dscaling.py ──▶ PNG
```

## 0. Prerequisites and environments

Reproduction spans two dependency sets, which trips people up:

- **The benchmark sweep** runs inside ann-benchmarks, which is Docker-based. It
  needs ann-benchmarks' own Python deps (the `docker` SDK, etc.) **and a running
  Docker daemon**. These are *not* in this package's `environment.yml`.
- **Tuning and figures** (this repo's `tuning/` and `reproduce/`) only need this
  package's deps (`numpy/scipy/scikit-learn/h5py/matplotlib/optuna/pyyaml`).

Simplest setup -- one environment that does everything:

```bash
# Set up ann-benchmarks per its own README (clone + install its requirements),
# then install this package into that same environment:
git clone https://github.com/erikbern/ann-benchmarks
cd ann-benchmarks && pip install -r requirements.txt && cd ..
pip install -e /path/to/MultiProbeANN[repro]

export ANN_BENCHMARKS_DIR="$PWD/ann-benchmarks"
```

Also install [Docker](https://docs.docker.com/get-docker/) and start the daemon.
(If you only want to play with the algorithm, run the tests, or re-plot from
existing results, the standalone `conda env create -f environment.yml` env is
enough -- you just can't run the sweep from it. `setup.sh` checks for the `docker`
module and a running daemon and tells you if either is missing.)

Integrate multiprobe and build the algorithm images (run from this repo, then the
clone):

```bash
./ann_benchmarks_integration/setup.sh "$ANN_BENCHMARKS_DIR"   # builds ann_multiprobe image

cd "$ANN_BENCHMARKS_DIR"
python install.py --algorithm voyager
python install.py --algorithm pynndescent
python install.py --algorithm annoy
python install.py --algorithm faiss
python install.py --algorithm bruteforce
```

## 1. Derive the multiprobe Pareto configs (optional but needed for a full front)

The shipped `config.yml` has a single operating point, which traces only a partial
multiprobe curve. To reproduce the full Pareto front, regenerate the sweep:

```bash
# Edit configs/glove200_search.yaml if you want fewer trials for a quick pass.
python tuning/optuna_search.py --config configs/glove200_search.yaml
python tuning/extract_pareto_configs.py \
    --db results/studies/glove200.db \
    --study glove200_multiprobe_pareto \
    --output ann_benchmarks_integration/config.yml
./ann_benchmarks_integration/setup.sh "$ANN_BENCHMARKS_DIR"   # re-copy the new config
```

NSGA-II is seeded (`optuna.seed` in the config) for reproducibility; change or
remove the seed to explore other fronts.

## 2. Run the benchmark sweep

```bash
./reproduce/run_benchmarks.sh "$ANN_BENCHMARKS_DIR"
```

This benchmarks all algorithms on glove-25/50/100/200-angular (covers Fig 1's
glove-200 and Fig 2b's full GloVe family). It is the long step -- single-CPU Docker
runs at up to 1.18M points take hours.

## 3. Render the figures

```bash
python reproduce/fig1_pareto.py     --ann-benchmarks-dir "$ANN_BENCHMARKS_DIR"
python reproduce/fig2b_dscaling.py  --ann-benchmarks-dir "$ANN_BENCHMARKS_DIR"
```

`reproduce/reference/d_scaling_exponents.csv` holds the paper's d-scaling exponent
per algorithm (multiprobe ≈ -0.92, the baselines ≈ -1.4 to -1.6) so you can
sanity-check the Fig 2b crossover against our numbers.

## Caveats

- Absolute QPS is hardware-dependent and will not match the paper. Multiprobe is a
  Python proof-of-concept, so its absolute throughput is conservative; the *trends*
  (log-linear Pareto, near-constant d-scaling vs steepening baselines) are the result.
- With only the shipped single-point `config.yml` (skipping step 1), the multiprobe
  curve in Fig 1 will be sparse -- run the tuning step for the full front.
- ann-benchmarks kills each algorithm container after a per-run timeout, default
  7200 s (2 hours; see `run.py --timeout` and `ann_benchmarks/runner.py`). On large
  datasets a near-full-scan config (high `n_probe` on a small grid, e.g. the
  glove-200 `d6-g5`/`d6-g2` points at the top probe counts) can exceed it. When that
  happens the container is killed and the harness silently moves on, dropping those
  highest-recall points from the front. The figure still renders, just without its
  high-recall tip. Raise the limit for the full sweep:

  ```bash
  python run.py --algorithm ann-multiprobe --dataset glove-200-angular --runs 1 --timeout 21600
  ```

  Use `--timeout -1` to disable the limit entirely (what the paper runs used).
