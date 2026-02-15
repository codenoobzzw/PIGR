# PGB: Post-Refinement for ANN Graph Indices

This repository contains the refactored implementation of our post-refinement pipeline for Approximate Nearest Neighbor (ANN) graph indices.
It supports three base graph families:

- HNSW
- NSG
- HCNNG

The codebase is organized for reproducible experiments and paper-style evaluation on SIFT-128.

## 1. Overview

The pipeline takes a pre-built ANN graph index and performs post-refinement to improve search quality under controlled runtime overhead.

Core features:

- Unified runner interface for HNSW / NSG / HCNNG
- Configurable refinement hyper-parameters (e.g., pruning ratio, core-k, efq schedule)
- Consistent CSV logging for analysis (`out/*.csv`)
- Standalone tool to build initial indices (`bin/build_indices`)

## 2. Repository Structure

```text
pgb_github_refactor_v2/
├─ CMakeLists.txt
├─ README.md
├─ anns-data/                  # bundled dataset conversion utilities
├─ include/
│  ├─ graph/                   # base ANN graph structures
│  └─ pgb/                     # refiner interface + implementation
├─ src/
│  ├─ build_indices.cpp
│  ├─ run_hnsw_final.cpp
│  ├─ run_nsg_final.cpp
│  └─ run_hcnng_final.cpp
├─ data/                       # runtime datasets (generated locally)
├─ indices/                    # runtime indices (generated locally)
└─ out/                        # runtime logs (generated locally)
```

## 3. Requirements

- Linux
- CMake >= 3.16
- C++17 compiler (GCC/Clang)
- OpenMP
- Python 3.8+ (for dataset conversion)
- Python packages: `numpy`, `h5py`

Install Python dependencies:

```bash
python3 -m pip install numpy h5py
```

## 4. Build

From repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Generated executables (in `bin/`):

- `run_hnsw_final`
- `run_nsg_final`
- `run_hcnng_final`
- `build_indices`

## 5. Dataset Preparation (SIFT-128)

Use bundled `anns-data` tools:

```bash
cd anns-data
python3 create_dataset.py --dataset sift-128-euclidean
cd ..
```

Copy generated files to runtime data directory:

```bash
mkdir -p data
cp anns-data/sift-128-euclidean.train.fvecs data/
cp anns-data/sift-128-euclidean.test.fvecs data/
cp anns-data/sift-128-euclidean.gt.ivecs data/
```

Expected files:

- `data/sift-128-euclidean.train.fvecs`
- `data/sift-128-euclidean.test.fvecs`
- `data/sift-128-euclidean.gt.ivecs`

## 6. Build Base Indices

```bash
./bin/build_indices \
  --base data/sift-128-euclidean.train.fvecs \
  --out-root indices/sift-128-euclidean \
  --threads 24 \
  --hnsw-m 16 --hnsw-efc 96 \
  --nsg-r 64 --nsg-efc 256 \
  --hcnng-s 7 --hcnng-t 15 --hcnng-ls 1250
```

Default outputs:

- `indices/sift-128-euclidean/hnsw/M_16_efc_96.idx`
- `indices/sift-128-euclidean/nsg/R_64_efc_256.idx`
- `indices/sift-128-euclidean/hcnng/s_7_T_15_Ls_1250.idx`

## 7. Run Post-Refinement

Run from repository root:

```bash
./bin/run_hnsw_final
./bin/run_nsg_final
./bin/run_hcnng_final
```

CSV logs are written to:

- `out/hnsw_final.csv`
- `out/nsg_final.csv`
- `out/hcnng_final.csv`

## 8. Reproducibility Notes

- Keep dataset, index parameters, and thread settings fixed for fair comparison.
- Current default runner settings target SIFT-128.
- Key runtime paths and knobs can be adjusted in:
  - `src/run_hnsw_final.cpp`
  - `src/run_nsg_final.cpp`
  - `src/run_hcnng_final.cpp`

## 9. Common Issues

1. `Error opening file: data/...`
   - Run binaries from repository root, or update path settings in runner files.

2. OpenMP not found during CMake configure
   - Install OpenMP toolchain and rerun CMake configure.

3. Output CSV not found
   - Check `cfg.log_csv` path in the corresponding runner and ensure `out/` is writable.
