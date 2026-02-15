# PGB Refactor v2

Refactored post-refinement pipelines for ANN graph indices (HNSW / NSG / HCNNG), with a reproducible layout for paper code release.

## 1. Build

From repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Executables are generated in `bin/`:

- `bin/run_hnsw_final`
- `bin/run_nsg_final`
- `bin/run_hcnng_final`
- `bin/build_indices`

## 2. Dataset Preparation (use bundled `anns-data`)

This repository already includes `anns-data/` from the original project. Use it directly.

Install dependencies:

```bash
python3 -m pip install numpy h5py
```

Generate SIFT vecs data (run inside `anns-data/`):

```bash
cd anns-data
python3 create_dataset.py --dataset sift-128-euclidean
```

Then place files into the expected runtime path:

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

## 3. Build Indices

```bash
./bin/build_indices \
  --base data/sift-128-euclidean.train.fvecs \
  --out-root indices/sift-128-euclidean \
  --threads 24 \
  --hnsw-m 16 --hnsw-efc 96 \
  --nsg-r 64 --nsg-efc 256 \
  --hcnng-s 7 --hcnng-t 15 --hcnng-ls 1250
```

Generated files:

- `indices/sift-128-euclidean/hnsw/M_16_efc_96.idx`
- `indices/sift-128-euclidean/nsg/R_64_efc_256.idx`
- `indices/sift-128-euclidean/hcnng/s_7_T_15_Ls_1250.idx`

## 4. Run

Run from repository root:

```bash
./bin/run_hnsw_final
./bin/run_nsg_final
./bin/run_hcnng_final
```

Outputs are written to `out/`.

## 5. What to upload to GitHub

Upload source and configs:

- `CMakeLists.txt`
- `include/`
- `src/`
- `scripts/`
- `anns-data/`
- `README.md`
- `.gitignore`

Do **not** upload generated artifacts and large runtime files:

- `build/`
- `bin/`
- `out/`
- `data/`
- `indices/`

## 6. Notes

- If you use another dataset/index name, update paths in `src/run_hnsw_final.cpp`, `src/run_nsg_final.cpp`, `src/run_hcnng_final.cpp`.
- OpenMP is required at build time.
