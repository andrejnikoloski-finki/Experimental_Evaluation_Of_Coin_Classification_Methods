# Coin Classification Experiments

Source code for the comparison of four coin classification methods on combined obverse and reverse images from the Corpus Nummorum (CN) database.

# Authors

Andrej Nikoloski

Milena Trajkoska

Biljana Risteska Stojkoska

# Publication

This repository contains the source code accompanying the paper Experimental Evaluation of Coin Classification Methods using Deep Learning Methods, accepted for presentation at TELFOR 2026.

## Dataset

All experiments use images from the Corpus Nummorum (CN)
Coin Image Dataset, publicly available on Zenodo:

https://doi.org/10.5281/zenodo.10033993

The dataset_splits directory contains the exact coin IDs, image filenames, original and merged class labels, and fixed training, validation, and test assignments used in all six experimental settings.

The original images can be downloaded from Zenodo.

## Experimental settings

| Source directory | Experiments | Coins | Original / merged classes |
| --- | --- | ---: | ---: |
| `coin_experiments_balanced/` | E1, E2 | 840 | 28 / 26 |
| `coin_experiments_imbalanced_min30/` | E3, E4 | 1,602 | 28 / 26 |
| `coin_experiments_coin_min20/` | E5, E6 | 3,072 | 90 / 88 |

The balanced dataset contains 30 specimens per original type. The second campaign uses all accepted specimens from the same 28 types. The third extends the task to eligible types with at least 20 accepted specimens. Types 3314/3987 and 20178/20184 are merged in the merged settings. Both label settings use the same images and fold assignments within each campaign.

## Requirements

Python 3.11 and the dependencies listed in each `requirements.txt`. The recorded cluster environment used Python 3.11.12, Singularity, an NVIDIA A100 GPU, CUDA 12.6 and the following package versions:

| Library | Version |
| --- | --- |
| PyTorch | 2.7.0+cu126 |
| torchvision | 0.22.0+cu126 |
| NumPy | 2.2.5 |
| SciPy | 1.17.1 |
| scikit-learn | 1.9.1 |
| OpenCV (headless) | 4.11.0.86 |
| Pillow | 11.0.0 |
| joblib | 1.6.0 |

Install a compatible PyTorch/torchvision pair for the target CUDA environment. Use either OpenCV package variant, not both. SIFT–BoVW–SVM runs on CPU; the neural models use GPU for the reported experiments. ImageNet pretrained weights must be available (download or local cache).

## Configurations

| Method | Candidate configurations and parameters |
| --- | --- |
| SIFT–BoVW–SVM | 256 x 256 grayscale, 100 visual words, MiniBatchKMeans (batch size 2048, n_init 1), L1 histogram normalization, StandardScaler, RBF SVM (C = 1 or 10, gamma = scale) |
| ResNet50 | ImageNet V2 weights; frozen backbone or layer4 fine-tuning; dropout 0.3 |
| Siamese ResNet18 | ImageNet V1 weights; frozen backbone or layer4 fine-tuning; 128-dimensional L2-normalized embeddings, contrastive loss (margin 1.0), 600 newly sampled pairs per epoch (equal same/different-type pairs, uniform class sampling), nearest training-class centroid |
| EfficientNet-B0 | ImageNet V1 weights; frozen backbone or last two feature blocks fine-tuned; dropout 0.3 |

Shared settings (see `config.py`):

- Selection seed: 42. Cross-validation seeds: 42, 123, 2026.
- Three repetitions of 5-fold stratified evaluation (approximately 70% train, 10% validation, 20% test).
- Neural input: 224 x 224 RGB, ImageNet normalization; training rotation ±15°, brightness and contrast jitter 0.15; no horizontal flips.
- Adam optimizer: head learning rate 0.001, unfrozen backbone learning rate 0.0001; batch size 32.
- Maximum 100 epochs, early stopping patience 12, minimum improvement 0.001. Candidates are selected using validation macro-F1; only the selected candidate is tested.
- Accuracy, macro-F1, weighted-F1 and per-class metrics; mean and sample SD over 15 folds, corrected paired tests with Holm adjustment (alpha 0.05). Predictions, confusion matrices and timing are saved.

The six settings comprise 720 candidate configurations and 360 final test evaluations.

## Running the experiments

Use the original combined CN images arranged by numeric type folders. Commands below run from the `coin_classification/` directory, with the required environment active. The expanded campaigns depend on earlier prepared datasets; run them in order. The dataset and generated outputs are not included in this source archive.

### E1 and E2

Place the original balanced preparation input in `dataset_types/` at the project root (or pass its actual location). Use the original dataset and class selection to reproduce the reported cohort.

```bash
python coin_experiments_balanced/prepare_dataset.py --dataset dataset_types --output experiments_28 --confirm-combined
python coin_experiments_balanced/run_experiments.py --output experiments_28 --device cuda --preflight-only --check-device
python coin_experiments_balanced/run_experiments.py --output experiments_28 --device cuda --confirm-audit
python coin_experiments_balanced/analyze_results.py --output experiments_28
```

### E3 and E4

The `pack_full_dataset.py` utility packages all raw images in the original 28 type folders. Use the full source dataset, not only the selected balanced images. Extract its archive into the project root to populate `coin_experiments_imbalanced_min30/dataset_types/`.

```bash
python coin_experiments_imbalanced_min30/pack_full_dataset.py --dataset /path/to/full/dataset_types --zip min30_data.zip

cd coin_experiments_imbalanced_min30
python coin_experiments/prepare_imbalanced.py --dataset dataset_types --baseline ../experiments_28 --output experiments_imbalanced --confirm-combined
python coin_experiments/run_experiments.py --output experiments_imbalanced --device cuda --confirm-audit
python coin_experiments/analyze_results.py --output experiments_imbalanced
python coin_experiments/compare_baseline.py --baseline ../experiments_28 --output experiments_imbalanced
cd ..
```

### E5 and E6

Run `pack_min20.py` on the full original source to obtain the eligible folders and the cross-type `conflicting_ids.txt` file. Extract its archive into the project root. The final conflict correction is already included in `prepare_min20.py`.

```bash
python coin_experiments_coin_min20/pack_min20.py --dataset /path/to/full/dataset_types --zip min20_data.zip

cd coin_experiments_coin_min20
python coin_experiments/prepare_min20.py --dataset dataset_types --baseline ../coin_experiments_imbalanced_min30/experiments_imbalanced --output experiments_min20 --conflicts conflicting_ids.txt --confirm-combined
python coin_experiments/run_experiments.py --output experiments_min20 --device cuda --confirm-audit
python coin_experiments/analyze_results.py --output experiments_min20
cd ..
```

Inspect the generated class selection and audit files before passing `--confirm-audit`. Each preparation requires a fresh output directory.

## Cluster and transfer utilities

For E3–E6, the included Slurm launchers are submitted from their respective campaign directory, after preparation. Create `logs/` before submission. The launchers use `COIN_ENV_ROOT` for the Singularity image (`coin_cuda126.sif`), Python environment (`container_venv/`) and cached weights (`pretrained_weights/`); it defaults to `$HOME/coin_finki`, the original runtime location. For example, from the Min30 campaign directory:

```bash
mkdir -p logs
export COIN_ENV_ROOT="$HOME/coin_finki"
sbatch train_imbalanced.slurm
```

For Min20, submit `train_min20.slurm` instead from its own directory. The cluster partition and GPU resource settings may require adjustment on another cluster.

The root-level `package.py` transfers freshly prepared balanced data and its matching code:

```bash
python package.py --project coin_experiments_balanced --prepared experiments_28 --zip balanced_transfer.zip
```

Extract that archive into the parent directory of `coin_classification/`, then run `python relocate.py` from its project root before training. The relocation helper updates the prepared dataset path only and refuses to work after training has started.
