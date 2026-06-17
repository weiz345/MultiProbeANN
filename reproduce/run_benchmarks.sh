#!/usr/bin/env bash
# Run the benchmark sweeps behind Fig 1 (glove-200-angular Pareto) and
# Fig 2b (d-scaling across glove-25/50/100/200-angular).
#
# Prereqs: build the algorithm Docker images in your ann-benchmarks clone first:
#   python install.py --algorithm ann_multiprobe
#   python install.py --algorithm voyager
#   python install.py --algorithm pynndescent
#   python install.py --algorithm annoy
#   python install.py --algorithm faiss
#   python install.py --algorithm bruteforce
# The GloVe HDF5s auto-download from ann-benchmarks.com on first run.
#
# This sweeps the multiprobe operating point shipped in config.yml. For the full
# multiprobe Pareto front, first regenerate config.yml via tuning/ (see
# docs/REPRODUCTION.md).
#
# Usage:
#   ./reproduce/run_benchmarks.sh /path/to/ann-benchmarks
#   (or set ANN_BENCHMARKS_DIR and run with no argument)

set -uo pipefail

ANN_BENCH="${1:-${ANN_BENCHMARKS_DIR:-}}"
if [[ -z "$ANN_BENCH" ]]; then
    echo "usage: $0 <path-to-ann-benchmarks>  (or set ANN_BENCHMARKS_DIR)"
    exit 1
fi
cd "$ANN_BENCH"

DATASETS="glove-25-angular glove-50-angular glove-100-angular glove-200-angular"

for DS in $DATASETS; do
    echo "=== $DS — $(date) ==="
    python run.py --algorithm ann-multiprobe --dataset "$DS" --runs 1 --timeout 7200 || true
    python run.py --algorithm voyager        --dataset "$DS" --runs 3 --timeout 7200 || true
    python run.py --algorithm pynndescent    --dataset "$DS" --runs 3 --timeout 7200 || true
    python run.py --algorithm annoy          --dataset "$DS" --runs 3 --timeout 7200 --run-disabled || true
    python run.py --algorithm faiss-ivf      --dataset "$DS" --runs 3 --timeout 7200 --run-disabled || true
    python run.py --algorithm bruteforce     --dataset "$DS" --runs 3 --timeout 7200 --run-disabled || true
done

echo ""
echo "done. generate figures from the MultiProbeANN repo:"
echo "  python reproduce/fig1_pareto.py   --ann-benchmarks-dir $ANN_BENCH"
echo "  python reproduce/fig2b_dscaling.py --ann-benchmarks-dir $ANN_BENCH"
