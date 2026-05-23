# Dataset Preparation

Large datasets and generated indexes are intentionally not committed. Keep local datasets under `data/` and generated indexes under `indices/`.

## Default SIFT Layout

The default runners expect these files:

```text
data/sift-128-euclidean.train.fvecs
data/sift-128-euclidean.test.fvecs
data/sift-128-euclidean.gt.ivecs
```

Default generated indexes are expected at:

```text
indices/sift-128-euclidean/hnsw/M_16_efc_96.idx
indices/sift-128-euclidean/nsg/R_64_efc_256.idx
indices/sift-128-euclidean/hcnng/s_7_T_15_Ls_1250.idx
```

## Preparing SIFT1M

Install Python dependencies:

```bash
python3 -m pip install numpy h5py
```

Run:

```bash
./scripts/prepare_sift.sh
```

This script runs `anns-data/create_dataset.py --dataset sift-128-euclidean` and copies the generated vecs files into `data/`.

You can also run the underlying command manually:

```bash
cd anns-data
python3 create_dataset.py --dataset sift-128-euclidean
cd ..
mkdir -p data
cp anns-data/data/sift-128-euclidean.train.fvecs data/
cp anns-data/data/sift-128-euclidean.test.fvecs data/
cp anns-data/data/sift-128-euclidean.gt.ivecs data/
```

## Building Default Indexes

After placing SIFT files under `data/`, build the default graph indexes:

```bash
./bin/build_indices \
  --base data/sift-128-euclidean.train.fvecs \
  --out-root indices/sift-128-euclidean \
  --threads 24 \
  --hnsw-m 16 --hnsw-efc 96 \
  --nsg-r 64 --nsg-efc 256 \
  --hcnng-s 7 --hcnng-t 15 --hcnng-ls 1250
```

## Paper Datasets

The paper uses SIFT1M, DEEP1M, DBPEDIA1M, MNIST, and AUDIO. The repository contains SIFT-oriented runner defaults and dataset utilities under `anns-data/`. Other datasets require external downloads or conversion steps and may require adapting the runner source files to point to the desired local vecs files and indexes.

External datasets have their own licenses, terms, and citation requirements. Verify those terms before downloading, converting, or redistributing any data.
