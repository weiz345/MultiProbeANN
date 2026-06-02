#!/usr/bin/env bash
set -euo pipefail

# Setup script for integrating ann-multiprobe into ann-benchmarks.
#
# On ARM64 / Apple Silicon, the upstream Dockerfiles for hnswlib, faiss,
# and faiss_hnsw won't build because they download x86-only binaries.
# This script detects the architecture and patches those Dockerfiles
# so that all algorithms build natively.
#
# Usage:
#   ./ann_benchmarks_integration/setup.sh /path/to/ann-benchmarks
#
# This copies the integration files, optionally patches ARM64-incompatible
# Dockerfiles, builds the Docker image, and verifies the setup is ready to run.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTEGRATION_DIR="$SCRIPT_DIR"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <path-to-ann-benchmarks>"
    echo "Example: $0 ../ann-benchmarks"
    exit 1
fi

ANN_BENCH="$(cd "$1" && pwd)"
TARGET_DIR="$ANN_BENCH/ann_benchmarks/algorithms/ann_multiprobe"
ALGO_DIR="$ANN_BENCH/ann_benchmarks/algorithms"

# Verify both repos exist
if [[ ! -f "$ANN_BENCH/run.py" ]]; then
    echo "Error: $ANN_BENCH does not look like an ann-benchmarks repo (run.py not found)"
    exit 1
fi

if [[ ! -d "$INTEGRATION_DIR" ]]; then
    echo "Error: $INTEGRATION_DIR not found"
    exit 1
fi

# ── Architecture detection ───────────────────────────────────────────
ARCH="$(uname -m)"   # aarch64 or arm64 on ARM; x86_64 on Intel
IS_ARM=false
if [[ "$ARCH" == "arm64" || "$ARCH" == "aarch64" ]]; then
    IS_ARM=true
fi
echo "Detected architecture: $ARCH (ARM=$IS_ARM)"

# ── Patch Dockerfiles for ARM64 ─────────────────────────────────────
# The upstream hnswlib, faiss, and faiss_hnsw Dockerfiles download
# x86-only Anaconda or build from an old repo.  On ARM we replace them
# with pip-based installs that have native wheels.
patch_dockerfile() {
    local file="$1"
    local content="$2"
    local backup="${file}.x86_backup"

    if [[ -f "$backup" ]]; then
        echo "  $file already patched (backup exists), skipping"
        return
    fi

    cp "$file" "$backup"
    printf '%s\n' "$content" > "$file"
    echo "  Patched $file (original saved to ${backup##*/})"
}

if $IS_ARM; then
    echo ""
    echo "ARM64 detected — patching upstream Dockerfiles for native builds ..."

    # hnswlib: replace source build with pip wheel
    if [[ -f "$ALGO_DIR/hnswlib/Dockerfile" ]]; then
        patch_dockerfile "$ALGO_DIR/hnswlib/Dockerfile" \
'FROM ann-benchmarks

RUN pip install --no-cache-dir hnswlib

RUN python -c '"'"'import hnswlib'"'"''
    fi

    # faiss: replace x86 Anaconda + conda install with pip wheel
    if [[ -f "$ALGO_DIR/faiss/Dockerfile" ]]; then
        patch_dockerfile "$ALGO_DIR/faiss/Dockerfile" \
'FROM ann-benchmarks

RUN pip install --no-cache-dir faiss-cpu

RUN python3 -c '"'"'import faiss; print(faiss.IndexFlatL2)'"'"''
    fi

    # faiss_hnsw: same fix as faiss
    if [[ -f "$ALGO_DIR/faiss_hnsw/Dockerfile" ]]; then
        patch_dockerfile "$ALGO_DIR/faiss_hnsw/Dockerfile" \
'FROM ann-benchmarks

RUN pip install --no-cache-dir faiss-cpu

RUN python3 -c '"'"'import faiss; print(faiss.IndexFlatL2)'"'"''
    fi
else
    echo "x86_64 detected — no Dockerfile patches needed"
fi

# ── Copy ann-multiprobe integration files ────────────────────────────
echo ""
echo "Copying integration files to $TARGET_DIR ..."
mkdir -p "$TARGET_DIR"
cp "$INTEGRATION_DIR/module.py"   "$TARGET_DIR/"
cp "$INTEGRATION_DIR/config.yml"  "$TARGET_DIR/"
cp "$INTEGRATION_DIR/Dockerfile"  "$TARGET_DIR/"
echo "  Copied module.py, config.yml, Dockerfile"

# ── Build Docker image ───────────────────────────────────────────────
echo ""
echo "Building Docker image (this may take a few minutes on first run) ..."
cd "$ANN_BENCH"
python3 install.py --algorithm ann_multiprobe

echo ""
echo "Setup complete. To run benchmarks:"
echo "  cd $ANN_BENCH"
echo "  python run.py --algorithm ann-multiprobe --dataset fashion-mnist-784-euclidean"
echo "  python plot.py --dataset fashion-mnist-784-euclidean -Y log"
