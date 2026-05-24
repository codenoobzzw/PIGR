#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DOC_HINT="See docs/DATA.md for dataset and index preparation instructions."

die() {
  echo "ERROR: $*" >&2
  echo "$DOC_HINT" >&2
  exit 1
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || die "Missing required file: $path"
}

require_executable() {
  local path="$1"
  [[ -x "$path" ]] || die "Missing required executable: $path. Run ./scripts/build.sh first."
}

for binary in bin/build_indices bin/run_hnsw_final bin/run_nsg_final bin/run_hcnng_final; do
  require_executable "$binary"
done

require_file "data/sift-128-euclidean.train.fvecs"
require_file "data/sift-128-euclidean.test.fvecs"
require_file "data/sift-128-euclidean.gt.ivecs"

RUNNERS="${PIGR_SANITY_RUNNERS:-hnsw}"
[[ -n "$RUNNERS" ]] || die "PIGR_SANITY_RUNNERS is empty."

mkdir -p out

expected_csv() {
  local base_csv="$1"
  echo "${base_csv%.csv}_prune0.020_jump8_repeat1.csv"
}

run_one() {
  local runner="$1"
  case "$runner" in
    hnsw)
      require_file "indices/sift-128-euclidean/hnsw/M_16_efc_96.idx"
      ./bin/run_hnsw_final
      require_file "$(expected_csv "out/hnsw_final.csv")"
      ;;
    nsg)
      require_file "indices/sift-128-euclidean/nsg/R_64_efc_256.idx"
      ./bin/run_nsg_final
      require_file "$(expected_csv "out/nsg_final.csv")"
      ;;
    hcnng)
      require_file "indices/sift-128-euclidean/hcnng/s_7_T_15_Ls_1250.idx"
      ./bin/run_hcnng_final
      require_file "$(expected_csv "out/hcnng_final.csv")"
      ;;
    *)
      die "Unknown runner '$runner'. Use hnsw, nsg, hcnng, or a space-separated combination."
      ;;
  esac
}

for runner in $RUNNERS; do
  echo "Running PIGR sanity path: $runner"
  run_one "$runner"
done

echo "PIGR sanity run completed. CSV outputs are under out/."
