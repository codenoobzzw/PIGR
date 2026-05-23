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

## Dataset Tooling

The main dataset entry point is:

```bash
cd anns-data
python3 create_dataset.py --dataset <DATASET_KEY>
```

For dense Euclidean datasets, the script writes:

```text
anns-data/data/<DATASET_KEY>.hdf5
anns-data/data/<DATASET_KEY>.train.fvecs
anns-data/data/<DATASET_KEY>.test.fvecs
anns-data/data/<DATASET_KEY>.gt.ivecs
```

Copy the generated vecs files into the repository-level `data/` directory before building indexes or running PIGR:

```bash
mkdir -p data
cp anns-data/data/<DATASET_KEY>.train.fvecs data/
cp anns-data/data/<DATASET_KEY>.test.fvecs data/
cp anns-data/data/<DATASET_KEY>.gt.ivecs data/
```

The conversion script computes ground truth with brute-force search. This can take substantial time and memory for 1M-scale datasets.

## Python Dependencies

Install Python dependencies:

```bash
python3 -m pip install numpy h5py scikit-learn psutil
```

DBPEDIA1M preparation additionally uses Hugging Face datasets and pandas:

```bash
python3 -m pip install datasets pandas
```

## Paper Dataset Keys

| Paper dataset | `anns-data` dataset key | Upstream/source used by current tooling | Notes |
| --- | --- | --- | --- |
| SIFT1M | `sift-128-euclidean` | TexMex SIFT archive from the IRISA FTP URL in `datasets_ann.py` | Default runner paths already target this dataset. |
| DEEP1M | `deep-image-96-euclidean` | `deep1M.tar.gz` from the CUHK GQR dataset URL in `datasets_ann.py` | The script expects `deep1M_base.fvecs` and `deep1M_query.fvecs` inside the archive. |
| DBPEDIA1M | `dbpedia-openai-1000k-euclidean` | Hugging Face dataset `KShivendu/dbpedia-entities-openai-1M` | Also supports 100k increments: `dbpedia-openai-100k-euclidean` through `dbpedia-openai-1000k-euclidean`. |
| MNIST | `mnist-784-euclidean` | Yann LeCun MNIST image gzip URLs in `datasets_ann.py` | Converts image vectors and computes Euclidean ground truth. |
| AUDIO | `audio-192-euclidean` | `audio.tar.gz` from the CUHK GQR dataset URL in `datasets_ann.py` | The script expects `audio_base.fvecs` and `audio_query.fvecs` inside the archive. |

## Preparing SIFT1M

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

## Preparing DEEP1M

```bash
cd anns-data
python3 create_dataset.py --dataset deep-image-96-euclidean
cd ..
mkdir -p data
cp anns-data/data/deep-image-96-euclidean.train.fvecs data/
cp anns-data/data/deep-image-96-euclidean.test.fvecs data/
cp anns-data/data/deep-image-96-euclidean.gt.ivecs data/
```

Generated repository-level files:

```text
data/deep-image-96-euclidean.train.fvecs
data/deep-image-96-euclidean.test.fvecs
data/deep-image-96-euclidean.gt.ivecs
```

## Preparing DBPEDIA1M

Install the additional dependencies first:

```bash
python3 -m pip install datasets pandas
```

Then run:

```bash
cd anns-data
python3 create_dataset.py --dataset dbpedia-openai-1000k-euclidean
cd ..
mkdir -p data
cp anns-data/data/dbpedia-openai-1000k-euclidean.train.fvecs data/
cp anns-data/data/dbpedia-openai-1000k-euclidean.test.fvecs data/
cp anns-data/data/dbpedia-openai-1000k-euclidean.gt.ivecs data/
```

Generated repository-level files:

```text
data/dbpedia-openai-1000k-euclidean.train.fvecs
data/dbpedia-openai-1000k-euclidean.test.fvecs
data/dbpedia-openai-1000k-euclidean.gt.ivecs
```

For smaller local checks, the current tooling also exposes `dbpedia-openai-100k-euclidean`, `dbpedia-openai-200k-euclidean`, and so on up to `dbpedia-openai-1000k-euclidean`.

## Preparing MNIST

```bash
cd anns-data
python3 create_dataset.py --dataset mnist-784-euclidean
cd ..
mkdir -p data
cp anns-data/data/mnist-784-euclidean.train.fvecs data/
cp anns-data/data/mnist-784-euclidean.test.fvecs data/
cp anns-data/data/mnist-784-euclidean.gt.ivecs data/
```

Generated repository-level files:

```text
data/mnist-784-euclidean.train.fvecs
data/mnist-784-euclidean.test.fvecs
data/mnist-784-euclidean.gt.ivecs
```

## Preparing AUDIO

```bash
cd anns-data
python3 create_dataset.py --dataset audio-192-euclidean
cd ..
mkdir -p data
cp anns-data/data/audio-192-euclidean.train.fvecs data/
cp anns-data/data/audio-192-euclidean.test.fvecs data/
cp anns-data/data/audio-192-euclidean.gt.ivecs data/
```

Generated repository-level files:

```text
data/audio-192-euclidean.train.fvecs
data/audio-192-euclidean.test.fvecs
data/audio-192-euclidean.gt.ivecs
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

For non-SIFT datasets, pass the matching `--base` file and choose an output root for that dataset. For example:

```bash
./bin/build_indices \
  --base data/mnist-784-euclidean.train.fvecs \
  --out-root indices/mnist-784-euclidean \
  --threads 24
```

The checked-in runner source files currently point to the default SIFT paths. To run PIGR on DEEP1M, DBPEDIA1M, MNIST, or AUDIO, update the corresponding `cfg.base_path`, `cfg.query_path`, `cfg.gt_path`, `cfg.index_path`, and `cfg.log_csv` values in `src/run_hnsw_final.cpp`, `src/run_nsg_final.cpp`, or `src/run_hcnng_final.cpp`.

## Paper Datasets

The paper uses SIFT1M, DEEP1M, DBPEDIA1M, MNIST, and AUDIO. The repository contains SIFT-oriented runner defaults and dataset utilities under `anns-data/`. Non-SIFT datasets require external downloads/conversion and adapting the runner source files to point to the desired local vecs files and indexes.

External datasets have their own licenses, terms, and citation requirements. Verify those terms before downloading, converting, or redistributing any data.
