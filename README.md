# PIGR: Post-hoc Iterative Graph Refinement

## Summary

PIGR is a post-hoc, trace-driven graph refinement framework for graph-based approximate nearest neighbor search. It refines existing graph indexes using self-query traces and budgeted edge edits without changing the deployed search procedure.

This artifact currently includes runners for these graph families:

- NSG
- HCNNG
- HNSW

## Paper

- Paper title: Beyond Structure-Driven Tuning: Cost-Aligned Graph Optimization for Approximate Nearest Neighbor Search
- Venue: KDD 2026
- Paper DOI: TODO: ACM paper DOI after RightsReview
- Artifact DOI: TODO: Zenodo DOI after GitHub release

## Artifact Scope

This repository contains source code, build instructions, dataset preparation notes, and scripts/documentation for quick sanity checks and paper-style experiments. Large public datasets, generated graph indexes, build products, logs, and full experiment outputs are not committed.

## Repository Structure

```text
.
|-- CMakeLists.txt              # CMake build configuration
|-- LICENSE                     # MIT License for this repository
|-- LICENSES.md                 # Third-party code and dependency notes
|-- CITATION.cff                # Citation metadata
|-- .zenodo.json                # Zenodo release metadata
|-- anns-data/                  # Dataset conversion utilities
|-- docs/
|   `-- DATA.md                 # Dataset sources and local layout
|-- include/
|   |-- graph/                  # Base ANN graph implementations
|   |-- ivf/                    # IVF helper code
|   |-- pigr/                   # PIGR refiner interfaces and implementations
|   |-- statistic/              # Evaluation helpers
|   `-- utils/                  # Binary IO, recall, timer, and utility helpers
|-- paper/
|   `-- README.md               # Notes for an optional extended report
|-- reproduce/
|   `-- README.md               # Reproduction workflow notes
|-- scripts/
|   |-- build.sh                # Release build helper
|   |-- prepare_sift.sh         # SIFT1M preparation wrapper
|   `-- run_sanity.sh           # Local sanity runner
`-- src/
    |-- build_indices.cpp       # Builds default base graph indexes
    |-- run_hcnng_final.cpp     # HCNNG PIGR runner
    |-- run_hnsw_final.cpp      # HNSW PIGR runner
    `-- run_nsg_final.cpp       # NSG PIGR runner
```

Runtime directories such as `data/`, `indices/`, `bin/`, `build/`, and `out/` are created locally and ignored by Git.

## Requirements

- Linux
- CMake >= 3.16
- C++17 compiler, such as GCC or Clang
- OpenMP
- Python 3.8+
- Python packages for dataset preparation, including `numpy` and `h5py`

Install Python dependencies as needed:

```bash
python3 -m pip install numpy h5py
```

## Build

From the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Or use the helper script:

```bash
./scripts/build.sh
```

Generated executables are written to `bin/`:

- `bin/build_indices`
- `bin/run_hcnng_final`
- `bin/run_hnsw_final`
- `bin/run_nsg_final`

## Dataset Preparation

The default checked-in runners expect SIFT1M-style files in vecs format:

```text
data/sift-128-euclidean.train.fvecs
data/sift-128-euclidean.test.fvecs
data/sift-128-euclidean.gt.ivecs
```

The bundled `anns-data` tooling can prepare the SIFT dataset:

```bash
./scripts/prepare_sift.sh
```

This wraps:

```bash
cd anns-data
python3 create_dataset.py --dataset sift-128-euclidean
```

and copies the generated vecs files into the repository-level `data/` directory.

The paper evaluates SIFT1M, DEEP1M, DBPEDIA1M, MNIST, and AUDIO. This repository includes helper tooling for SIFT-style preparation and additional dataset download/conversion utilities under `anns-data/`, but external datasets must be downloaded according to their own licenses and terms. See `docs/DATA.md` for expected file names and layout.

## Quick Sanity Test

Build the code first:

```bash
./scripts/build.sh
```

Prepare SIFT data or place compatible vecs files under `data/`, then build the default indexes:

```bash
./bin/build_indices \
  --base data/sift-128-euclidean.train.fvecs \
  --out-root indices/sift-128-euclidean \
  --threads 24 \
  --hnsw-m 16 --hnsw-efc 96 \
  --nsg-r 64 --nsg-efc 256 \
  --hcnng-s 7 --hcnng-t 15 --hcnng-ls 1250
```

Run the sanity script:

```bash
./scripts/run_sanity.sh
```

The script checks required binaries, data files, and indexes. If data or indexes are missing, it exits with a clear error and points to `docs/DATA.md`. By default it runs the HNSW sanity path and writes CSV output under `out/`, such as `out/hnsw_final.csv`.

To run multiple default runners:

```bash
PIGR_SANITY_RUNNERS="hnsw nsg hcnng" ./scripts/run_sanity.sh
```

## Reproducing Paper-Style Experiments

The default runner entry points are:

- HNSW: `bin/run_hnsw_final`
- NSG: `bin/run_nsg_final`
- HCNNG: `bin/run_hcnng_final`

Each runner configures dataset paths, index paths, logging paths, and PIGR refinement knobs in the corresponding `src/run_*_final.cpp` file. Full 1M-scale experiments may take substantial time and CPU resources. Keep dataset versions, index construction parameters, thread counts, and runner configurations fixed when comparing results.

See `reproduce/README.md` for quick, SIFT-based, and full paper-style workflows.

## Extended Technical Report

An optional extended technical report PDF may be placed at:

```text
paper/PIGR_extended_technical_report.pdf
```

No extended report PDF is currently included.

## Citation

Please cite the associated KDD 2026 paper and this software artifact. Citation metadata is provided in `CITATION.cff`.

## License

This repository is released under the MIT License. See `LICENSE` and `LICENSES.md`.

## Contact

For questions, please open a GitHub issue or contact the paper authors.
