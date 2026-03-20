# Dat675 project.

### The goal:

See if we could expand the BACE dataset from [MoleculeNet-BACE](https://moleculenet.org/datasets-1), using *synthethic data augmentation.*

### Structure of the project:

The project is divided into three distinct, but chained parts (which has its own more detailed READMEs).  The three parts are:

1. #### Pre-processing:

   1. Loads, cleans and canoicalize the data
   2. Splits the data into five CV-folds.
   3. Avoids data leakage
2. #### $\beta $-CVAE:

   1. Points towards the data from pre-processing
   2. Trains a $\beta$-CVAE, for each of the five folds.
   3. Generates (for each of the five CV-iterations) n-valid, molecules of which does not share scaffolds with the validation or test-sets.
      1. Also analyses the generated molecules and produce plots describing everything from t-SNE of the Morgan fingerprints, to the label head residual errors.
      2. Computes V.U.N metrics, and calculates the external Diversity between the generated molecules and the train dataset.
3. #### Downstream models

   1. MPNN/GAT (used synonomusly):
      - Training: three diffrent mixes of synthetic versus BACE data. (synthetic data matched with each CV-iteration). The mixes where: **0%,33%,67%**. The 0% is  a baseline Where the percentage describes what share of the data is synthetic vs BACE data. Each iteration has around 1000 BACE datasamples for training, and around 260 for validation.
      - The MPNN can also be pre-trained on the synthetic data (for each of the three datasets, 0%,33%,67%). The 0% case means that the model sees no Syntheitc data (BASELINE)  Then fine-tuned on only natrual data of each set.
   2. RF (Random forrest):
      - Trains on the mixes, and does a parameter sweep

---

## Setup:

### Requirements

- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
- Preferably a machine with [CUDA support](https://en.wikipedia.org/wiki/CUDA) for faster training

### Create and activate environment

From the project root:

If you have a CUDA 12.1 compatible NVIDIA GPU (typical Windows/Linux setup), use:

```powershell
conda env create -f environment.yml
conda activate rdkit_draw
```

If you are on a modern Mac (Apple Silicon, e.g. M1/M2/M3), use the Apple Silicon environment file:

```bash
conda env create -f environment.macos-arm64.yml
conda activate rdkit_draw_mac
```

Notes for Mac:

- Training will run on CPU or Apple MPS, not CUDA.
- The root `environment.yml` is Windows/CUDA-pinned and includes a Windows-specific `prefix`, so it is not portable to macOS as-is.

If environment creation fails because of a machine-specific `prefix:` in `environment.yml`, remove the `prefix` line and run the command again.

If activation fails in PowerShell:

```powershell
conda init powershell
```

Then restart terminal and activate again.

Optional update command (if environment already exists):

```powershell
conda env update -f environment.yml --prune
```

Quick package check:

```powershell
python -c "import torch, rdkit, sklearn, pandas, matplotlib; print('ENV_OK')"
```

---

## How to run the full pipeline

Run the three parts in this order:

1. Pre-processing
2. beta-CVAE generation
3. Downstream MPNN/GAT training + holdout evaluation

All commands below are run from the project root.

### 1) Pre-processing

This creates scaffold-split folds and the held-out test set.

```powershell
python src/original_preprocessing.py
```

Outputs:

- data/heldout_datasets/heldout_testset.csv
- data/combination_1300_molecules_and_0_%_synthetic/original_fold_0.csv ... original_fold_4.csv

### 2) beta-CVAE training and synthetic molecule generation

This trains the CV pipeline over folds and generates synthetic molecules.

Note: in `vae/fold_pipeline_config.example.yaml`, both `train.enabled` and `sampling.enabled` are `false` by default. Set them to `true` before running generation.

```powershell
python vae/run_fold_pipeline.py --config vae/fold_pipeline_config.example.yaml
```

Fast smoke run (single fold):

```powershell
python vae/run_fold_pipeline.py --config vae/fold_pipeline_config.example.yaml --only-fold 0
```

Expected synthetic outputs are written under:

- vae/fold_pipeline_outputs/cv_iteration_0 ... cv_iteration_4

### 3) Build mixed datasets (33% and 67% synthetic)

After beta-CVAE generation, create the mixed training folders:

```powershell
python src/synthetic_preprocessing.py
```

Outputs:

- data/combination_1950_molecules_and_33_%_synthetic
- data/combination_3900_molecules_and_67_%_synthetic

### 4) Downstream MPNN/GAT training

Train downstream GAT/MPNN models on 0%, 33%, and 67% datasets (5-fold CV each):

```powershell
python src/GAT_model/gat_predictor.py
```

Main artifacts:

- src/GAT_model/0%/checkpoints, src/GAT_model/33%/checkpoints, src/GAT_model/67%/checkpoints
- src/GAT_model/0%/MPNN_cv_results_0%.csv (and matching files for 33%/67%)

### 5) Held-out test evaluation for trained GAT/MPNN models

Evaluate saved checkpoints on heldout_testset.csv:

```powershell
python src/GAT_model/run_gat.py
```

Evaluate only one dataset (example 0%):

```powershell
python src/GAT_model/run_gat.py --folder 0%
```

Outputs:

- results/pretrain/gat_results_heldout.csv or results/no_pretrain/gat_results_heldout.csv
- results/pretrain/GAT_predictions_heldout_set.csv or results/no_pretrain/GAT_predictions_heldout_set.csv

---

## Regenerating figures/results for report

- Pre-processing and dataset files are regenerated by running the scripts above.
- VAE analysis plots are regenerated as part of the fold pipeline when analysis is enabled in the config.
- Downstream GAT/MPNN metrics and held-out predictions are regenerated by rerunning gat_predictor.py and run_gat.py.
