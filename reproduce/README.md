# Reproduction Notes

## Quick Sanity Run

From the repository root:

```bash
./scripts/build.sh
./scripts/run_sanity.sh
```

The sanity script checks required binaries, required data files, and the selected default index before running. If required files are missing, it exits with a clear message and points to `docs/DATA.md`.

By default, the sanity script runs the HNSW path. To run all default graph families:

```bash
PIGR_SANITY_RUNNERS="hnsw nsg hcnng" ./scripts/run_sanity.sh
```

Expected CSV outputs are written under `out/`, for example:

```text
out/hnsw_final.csv
out/nsg_final.csv
out/hcnng_final.csv
```

## SIFT-Based Run

Prepare SIFT data:

```bash
./scripts/prepare_sift.sh
```

Build default indexes:

```bash
./bin/build_indices \
  --base data/sift-128-euclidean.train.fvecs \
  --out-root indices/sift-128-euclidean \
  --threads 24 \
  --hnsw-m 16 --hnsw-efc 96 \
  --nsg-r 64 --nsg-efc 256 \
  --hcnng-s 7 --hcnng-t 15 --hcnng-ls 1250
```

Run the PIGR runners:

```bash
./bin/run_hnsw_final
./bin/run_nsg_final
./bin/run_hcnng_final
```

## Full Paper-Style Runs

Full experiments should use fixed dataset versions, fixed base-index construction parameters, fixed thread counts, and fixed runner configurations. The current runner entry points are:

- HNSW: `src/run_hnsw_final.cpp`
- NSG: `src/run_nsg_final.cpp`
- HCNNG: `src/run_hcnng_final.cpp`

Edit those files to point to local prepared datasets and indexes for SIFT1M, DEEP1M, DBPEDIA1M, MNIST, or AUDIO. Large 1M-scale runs may take substantial time and CPU resources.

## Caveats

This repository does not include large public datasets, generated indexes, build products, logs, or full experiment outputs. Runtime outputs belong in ignored local directories such as `data/`, `indices/`, and `out/`.
