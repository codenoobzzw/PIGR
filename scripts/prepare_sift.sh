#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python3}"

if [[ ! -f anns-data/create_dataset.py ]]; then
  echo "ERROR: anns-data/create_dataset.py was not found." >&2
  exit 1
fi

"$PYTHON_BIN" -c "import numpy, h5py" || {
  echo "ERROR: Python packages numpy and h5py are required." >&2
  echo "Install them with: $PYTHON_BIN -m pip install numpy h5py" >&2
  exit 1
}

(
  cd anns-data
  "$PYTHON_BIN" create_dataset.py --dataset sift-128-euclidean
)

mkdir -p data

for suffix in train.fvecs test.fvecs gt.ivecs; do
  src="anns-data/data/sift-128-euclidean.${suffix}"
  dst="data/sift-128-euclidean.${suffix}"
  if [[ ! -f "$src" ]]; then
    echo "ERROR: expected generated file not found: $src" >&2
    exit 1
  fi
  cp "$src" "$dst"
done

echo "SIFT vecs files are ready under data/."
