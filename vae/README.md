![Screenshot](figure.png)

# Conditional VAE for molecular generation (PyTorch)

Reference paper( for the code):

- https://jcheminf.biomedcentral.com/articles/10.1186/s13321-018-0286-7
- https://arxiv.org/abs/1806.05805
- 
- 

Reference papers for CVAE:

- Lim, J., Ryu, S., Kim, J.W. *et al.* Molecular generative model based on conditional variational autoencoder for de novo molecular design. *J Cheminform*  **10** , 31 (2018). https://doi.org/10.1186/s13321-018-0286-7

  - With github of: https://github.com/jaechanglim/CVAE

Reference paper for $\beta$-CVAE:

- Guang Jun, De Tao, Bingquan "Balancing Exploration and Exploitation:
  Disentangled β-CVAE in De Novo Drug Design" (aug 2023)
- https://arxiv.org/abs/2306.01683

Beta term can controll the entagleness of latent space. Making molecules more disentangeled.  of molecules (and stabilize them)

Reference paper for Vae with label prediction:

- "Automatic Chemical Design Using a Data-Driven
  Continuous Representation of Molecules" By Gómez-Bombarelli et al.
  Landmark paper. (2018)
- https://doi.org/10.1021/acscentsci.7b00572

This repository now contains an extended implementation that supports both:

- `lstm` CVAE (paper-style baseline), and
- `transformer` CVAE (an extension).

## What is different from the original paper implementation

 Modifications in this repo include:

(Its a bit of a ship of Theseus situation since so much is changed...)

- Changed from Tensorflow to Pytorch!
- Dual architecture switch in one `CVAE` class: `model_mode = lstm | transformer`.
- Saved training/model recreation config (`training_config.json`) during training.
- Sampling that can auto-load training config from the checkpoint folder (no manual architecture retyping).
- Added $\beta$-annealing to prevent posterior collapse.

  - So this is now a $\beta$-CVAE, based on works by [Nicholas Ang et al](https://arxiv.org/abs/2306.01683)  Who was based on [Higgints et al](https://www.cs.toronto.edu/~bonner/courses/2022s/csc2547/papers/generative/disentangled-representations/beta-vae,-higgins,-iclr2017.pdf).
  - Higher $\beta \implies$strenghtens constraints of latent space to be disentangled (traversable)
  - lower $\beta \implies$greater flexability in the representation.
- A bunch of "tricks of the trade" such as:

  - Lr adjustment on platue
  - dropouts (can prevent overfitting, and increase generalization)
  - weight decay and "adamw" optimizer
  - Ability to use both adam and adamW for optimizer
  - kl annealing holdout and warmup ( so as to disentangle latentspace, and help the unstable transformers.)
  - AMP for increased training speed.
  - Early stopping to prevent overfitting
  - Gaussian sampling
  - Augmentation by resampling (adding multiple instances of non-cannonical smiles for molecules)
    - This is helpfull for the base dataset.
    - 
- Modular config helpers in the `utils/` package for defaults, JSON load/save, and compose-from-overrides.
- Improved generation filtering/reporting in sampling (`unique`, `invalid`, `duplicates`, `in_training`).
- EOS-aware early stopping in decode loop for faster generation. So it does not go trough everything multiple times
- Ability to use only a subset of parameters for conditions compared to the origonal papers which had: MW,LogP, TPSA, HBD, HBA
- Latent memory injection into the Transformer-decoder. This is since the decoder produces sequences conditioned on both *z* and *c.* This means that for each time step, the decoder builds token input from: token embeddings, latent vector z and condition vector c, where both z and c is broadcasted across time steps. Then a memory vector is built and alastly cross-attention is applied in decoder. (a technique studied in the context of  LLMS for Memory injection atacks...)
- Separate prediction head for label predictions, this introduces a $\lambda_l$ term with is a label loss importance coeficent. It also introduces the label loss as **MSE**
- Label prediction head that samples on latent varible, and/or the target vector c. so: p(z,c)
- Oversampling by taking synonyms of the SMILES, to learn the underlying meaning instead of

## The ELBO optimization of $\beta$-CVAE:

$$
logp_\theta(x|z) \ge \mathcal{L}(\theta,\phi,x,z) = \underbrace{\mathbb{E}_{q_\phi(z|x)}[log\underbrace{p_\theta(x|z,c)}_{\text{Conditional likleyhood}}]}_{\text{Reconstruction error (decoder)}}-\beta\underbrace{ D_{KL}[\underbrace{q_\phi(z|x,c)}_{\text{Approximated posterior}}||\underbrace{p(z)}_{\text{prior}}]}_{D_{KL},\text{ Kullback-lieber term (encoder)}} + \underbrace{\lambda_l \mathcal{L}}_{\text{Label head loss (MSE)}}
$$

### In simpler terms:

$$
ELBO \ loss = \text{reconstruction loss} + \beta \times \text{KL (latent loss)} + \text{label loss weigt} \times \text{label loss}
$$

Where:

#### *Random varibles and dist....:*

* x *:* data, observations
* *c* : condition vector can be: LogP, MW.....
* *z*: Latent varible (possible molecule space)
* 

#### Parameters $\phi \ and \ \theta$:

###### $\theta$(decoder/generative parameters):

Parameters on the conditional likleyhood model:

$$
p_\theta(x|z,c)
$$

Decoder network. Givent latent (*z*) and condition *c:* outputs distribution over x. since smiles $\implies$

$$
p_\theta(x|z,c): \\ \text{Factorizes over timesteps as an autoregressice categorial distribution (softmax over tokens/atoms/smiles-letters)}
$$

###### $\phi$ (encoder/ variational parameters):

Tries to approximate the posterior:

$$
q_\phi(z|x,c)
$$

It outputs a distribution over latent varibles *z.*

In this project the prior is assumed to be (conditioned on *c*):

$q(z|x,c) \in {\mathcal{N}(\mu_\phi(x,c),diag(\sigma_\phi^2(x,c)))}$

One has to approximate the posterior since the true posterior: $p(z|x,c)$ is intractable since:

$$
p(z|x,c) = \frac{p(x|z,c)p(z|c)}{\underbrace{p(x|c)}_{\text{intractable}}}
$$

$$
\underbrace{p(x|c)}_{\text{Marginal likleyhood/Evidence }} = \int p(x|z,c) p(z|c)dz
$$

Which would mean having to find the probability of all possible real-latent varible *c*-values (impossible). And especially in the case of smiles where they are discrete....

So the encoder must approximate it, and the approximated posterior is denoted as: $q_\phi(z|x,c)$

## Label-Prediction head:

The labels will be sampeled from latent space: $f(z)$. This head is separate from the decoder. The label head can be toggled to take the target into account  . To be able to sucsessfully label the labels one needs an "disentangeled" latentspace. Disentangeled meaning that molecules sharing properties are in distinct "chunk like" areas of the latentspace. This is since an entangeled latentspace could mean that if one samples randomly, eventough the sample lies close to another from the training set, it does not nesecarily imply that they share features and labels (think of it like activitycliffs laying everywhere in propertyspace but this time in latentspace). And to actually disentangle latentspace one can use a larger  $\beta$ term. This is based on research done as mentioned by Guang Jun et al. The labels generated by the label head do have errors in pIC50, so they should be seen more as "fussy" than hard labels.

---

## BACE baseline workflow (pIC50-only conditioning)

## Pipeline Overview

This repository implements a complete **cross-validation (CV) drug discovery pipeline** for training conditional VAE models and generating novel molecules with desired properties. The pipeline orchestrates three tightly integrated stages: **training**, **molecular generation**, and **analysis-**all controlled by a single configuration file.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│  FoldPipeline: CV Orchestrator (run_fold_pipeline.py)              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  For each CV fold (fold_0, fold_1, ...):                            │
│                                                                      │
│  1. DATA SPLIT                                                       │
│     Validation Fold i  +  Training Folds (all others)               │
│          ↓                          ↓                                │
│     validation.txt             training.txt                         │
│                                                                      │
│  2. TRAINING STAGE (if train.enabled=true)                          │
│     ┌──────────────────────────────────────────────────────────┐    │
│     │ scripts/train_labels.py (via Python subprocess)         │    │
│     │                                                          │    │
│     │ Trains β-CVAE Model (LSTM or Transformer)              │    │
│     │ ├─ Encoder: q_φ(z|x,c) → latent vector z              │    │
│     │ ├─ Decoder: p_θ(x|z,c) → SMILES string               │    │
│     │ └─ Label Head: f(z,c) → property predictions           │    │
│     │                                                          │    │
│     │ Outputs: checkpoint.pt, training_config.json            │    │
│     └──────────────────────────────────────────────────────────┘    │
│                   checkpoint.pt                                      │
│                        ↓                                             │
│  3. SAMPLING STAGE (if sampling.enabled=true)                      │
│     ┌──────────────────────────────────────────────────────────┐    │
│     │ utils/sampling_pipeline_main.py                         │    │
│     │                                                          │    │
│     │ Generates molecules by:                                 │    │
│     │ ├─ Sample target properties (from mode: uniform/gauss)  │    │
│     │ ├─ Sample latent vectors z ~ N(0,I)                    │    │
│     │ ├─ Decode z+target to SMILES strings                   │    │
│     │ ├─ Filter via:                                          │    │
│     │ │  ├─ RDKit validity checks                             │    │
│     │ │  ├─ Exclude training/validation/heldout scaffolds     │    │
│     │ │  ├─ Property validity window checks                   │    │
│     │ │  └─ Duplicates & uniqueness constraints               │    │
│     │ └─ Compute molecular descriptors & metrics              │    │
│     │                                                          │    │
│     │ Outputs: generated.csv, quality_summary.csv             │    │
│     └──────────────────────────────────────────────────────────┘    │
│        generated.csv                                                 │
│             ↓                                                        │
│  4. ANALYSIS STAGE (if analysis.enabled=true)                      │
│     ┌──────────────────────────────────────────────────────────┐    │
│     │ analysis_modules/pipeline.py                            │    │
│     │                                                          │    │
│     │ Per-fold analysis:                                       │    │
│     │ ├─ Distribution plots (property ranges)                  │    │
│     │ ├─ Scaffold diversity analysis                           │    │
│     │ ├─ Tanimoto similarity histograms                        │    │
│     │ ├─ Prediction error plots (vs. labels)                  │    │
│     │ ├─ Chemical space visualizations (PCA/t-SNE)            │    │
│     │ └─ Summary statistics (V/U/N metrics)                    │    │
│     │                                                          │    │
│     │ Outputs: *.png plots, analysis_summary.json              │    │
│     └──────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ────────────────────────────────────────────────────────────        │
│  After all folds complete:                                           │
│                                                                      │
│  5. CROSS-FOLD AGGREGATION (if cv_combo.enabled=true)              │
│     ├─ Aggregate metrics across all CV iterations                  │
│     ├─ Combine V/U/N statistics                                    │
│     ├─ Generate ensemble plots & stats                             │
│     └─ Output: cv_combo_metrics_*.png, csv, json                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
vae/
├── run_fold_pipeline.py              <- Main entrypoint
├── fold_pipeline_config.example.yaml  <- Single config for entire pipeline
├── README.md                          <- This file
│
├── scripts/
│   ├── train_labels.py               <- Invoked for training stage
│   └── __init__.py
│
├── utils/
│   ├── run_fold_pipeline_main.py      <- CV orchestration logic
│   ├── sampling_pipeline_main.py      <- Molecule generation logic
│   ├── train_labels_main.py           <- Training entrypoint (called by scripts/)
│   ├── labels.py                      <- Sampling filters, quality metrics
│   ├── core.py                        <- Chemistry helpers, JSON/config I/O
│   ├── pipeline_helpers.py            <- Shared stats, VUN computation
│   └── __init__.py
│
├── analysis_modules/
│   ├── pipeline.py                    <- Per-fold analysis orchestration
│   ├── chem_utils.py                  <- RDKit wrappers (fingerprints, scaffolds)
│   ├── config.py                      <- Analysis config schema
│   ├── io_utils.py                    <- DataFrame loading for analysis
│   └── __init__.py
│
├── pipeline/
│   ├── fold_data.py                   <- CV fold discovery & data conversion
│   └── __init__.py
│
├── models/
│   ├── model_labels.py                <- CVAE model definition
│   └── __init__.py
│
├── fold_pipeline_outputs/             <- Generated outputs (per-fold + combined)
│   ├── cv_iteration_0/
│   │   ├── generated/
│   │   │   ├── generated.csv          <- Sampled molecules
│   │   │   └── quality_summary.csv    <- Sampling metrics
│   │   └── analysis/
│   │       ├── analysis_summary.json
│   │       ├── *.png                  <- Plots & visualizations
│   │       └── ...
│   ├── cv_combo/                      <- Cross-fold aggregation
│   │   ├── cv_combo_metrics_stats.json
│   │   └── *.png
│   └── cross_fold_analysis_summary.json
│
├── save/fold_pipeline_runs/           <- Training artifacts (models, logs)
│   └── cv_iteration_k/
│       └── training/
│           ├── model_best.ckpt-*.pt
│           ├── training_config.json
│           └── history.csv
│
├── graphics/                          <- Scratch/notebooks
├── audit_outputs/                     <- Code quality audits (optional)
└── fold_pipeline_config.example.yaml  <- Configuration (see below)
```

---

## Environment Setup (Conda)

This project is designed to run with the Conda environment defined in `environment.yml`.

### 1. Create the environment

From the project root (`vae/`):

```powershell
conda env create -f environment.yml
```

If you already have it and want to update packages from the file:

```powershell
conda env update -f environment.yml --prune
```

### 2. Activate the environment

```powershell
conda activate rdkit_draw
```

If activation fails in PowerShell, initialize conda for your shell and restart terminal:

```powershell
conda init powershell
```

Alternatively, run commands from Anaconda Prompt.

### 3. Verify key packages

```powershell
python -c "import torch, rdkit, sklearn, pandas, matplotlib; print('ENV_OK')"
```

### 4. Run the pipeline

```powershell
python run_fold_pipeline.py --config fold_pipeline_config.example.yaml
```

### 5. Run a single fold (faster smoke run)

```powershell
python run_fold_pipeline.py --config fold_pipeline_config.example.yaml --only-fold 0
```

---

## How to Run

### 1. Full CV Pipeline (Train + Sample + Analyze)

```bash
python run_fold_pipeline.py --config fold_pipeline_config.example.yaml
```

This runs all 5 CV folds (if data has 5 fold files) with training, sampling, and analysis enabled per fold, followed by cross-fold aggregation.

**Monitors:**

- Each stage logs to `fold_pipeline_outputs/cv_iteration_k/logs/`
- Training output goes to `save/fold_pipeline_runs/cv_iteration_k/training/`
- Per-fold results under `fold_pipeline_outputs/cv_iteration_k/{generated,analysis}/`

---

### 2. Single CV Fold Only

```bash
python run_fold_pipeline.py --config fold_pipeline_config.example.yaml --only-fold 0
```

Runs only fold 0 with training, sampling, and analysis. Useful for:

- Debugging a single iteration
- Quick validation on small datasets
- Testing configuration changes

---

### 3. Analysis-Only Mode (Reuse Existing Results)

Set in `fold_pipeline_config.example.yaml`:

```yaml
train:
  enabled: false
sampling:
  enabled: false
analysis:
  enabled: true
cv_combo:
  enabled: true
  only: false
```

Then run:

```bash
python run_fold_pipeline.py --config fold_pipeline_config.example.yaml
```

**What happens:**

- Skips training entirely
- Skips sampling entirely
- Loads existing `generated.csv` and `quality_summary.csv` from `fold_pipeline_outputs/cv_iteration_k/generated/`
- Runs analysis on saved molecules
- Aggregates results across all folds
- Ideal for: re-analyzing with different plot settings, generating new visualizations

---

### 4. Cross-Fold Aggregation Only

Set in `fold_pipeline_config.example.yaml`:

```yaml
cv_combo:
  enabled: true
  only: true
```

Then run:

```bash
python run_fold_pipeline.py --config fold_pipeline_config.example.yaml
```

**What happens:**

- Skips all per-fold processing
- Reads existing analysis summaries and generated molecules from each fold
- Produces combined plots, stats, and error analyses under `fold_pipeline_outputs/cv_combo/`

---

## Configuration Guide

The **`fold_pipeline_config.example.yaml`** is the single source of truth for the entire pipeline. Key sections:

### 1. Data & Paths

```yaml
workspace_root: .
training_output_root: save/fold_pipeline_runs
artifacts_output_root: fold_pipeline_outputs
train_validation_folds_dir: ../data/combination_1300_molecules_and_0_%_synthetic
fold_glob: "*fold_*.csv"
smiles_column: smiles
label_columns:
  - pIC50
```

- **fold_glob**: Pattern to find fold CSV files. File names must end with integer (e.g., `fold_0.csv`, `fold_iteration_2.csv`)
- Pipeline auto-discovers and creates one CV iteration per fold file

### 2. Training Configuration

```yaml
train:
  enabled: true
  base_config:
    model:
      mode: transformer          # lstm or transformer
      latent_size: 200
      unit_size: 256
      predict_labels: true       # Enable multi-task label head
      include_condition_in_label_head: true
      label_loss_weight: 1.0
    transformer:
      heads: 8
      dropout: 0.2
    optimization:
      optimizer: adamw
      lr: 1.0e-3
      weight_decay: 0.001
      grad_clip_norm: 4.0        # Prevents exploding gradients
      use_amp: false             # Automatic mixed precision
    training:
      batch_size: 64
      num_epochs: 120
      early_stopping_patience: 10
    kl:
      enabled: true
      max_beta: 4.0              # Beta-annealing for disentangled latent space
      warmup_epochs: 8
```

**Key hyperparameters:**

- `include_condition_in_label_head`: Whether label head sees condition *c* or only latent *z*
- `max_beta`: Controls disentanglement of latent space (higher means more disentangled)
- `weight_decay`: L2 regularization; prevents overfitting, this is good.
- `grad_clip_norm`: Gradient clipping norm; stabilizes transformer training
- `early_stopping_patience`: Stops if validation loss doesn't improve for N epochs

### 3. Sampling Configuration

```yaml
sampling:
  enabled: true
  target_sampling_mode: training_dist  # single_target, training_dist, uniform_range, uniform_range_strict
  num_unique: 10000                     # Target number of unique sampled molecules
  max_batches: 5000                     # Max generation batches before stopping
  temperature: 0.9                      # Sampling temperature (lower → more deterministic)
  top_k: 20                             # Top-k sampling parameter
  
  # For training_dist mode: sample targets from Gaussian around training mean/std
  training_dist_std_scale: 1.0
  training_dist_clip_n_std: 2.5        # Clip at +/- 2.5 
  
  # For uniform_range mode: sample uniformly in property range
  uniform_range: [3.0, 9.0]            # pIC50 range
  uniform_strict_bins: 20               # Split range into 20 bins for balanced sampling
  
  # Filtering & exclusion
  exclude_training: true                # Remove training set scaffolds from generated
  exclude_validation_scaffolds: true
  exclude_heldout_scaffolds: true       # Remove heldout dataset scaffolds
  heldout_smiles_csv: ../data/heldout_datasets/heldout_testset.csv
  
  # Chemistry filters
  strip_salts: true
  decharge: true
  canonicalize_tautomer: true
  max_heavy_atoms: 60
  require_neutral: true
  
  # Output columns in generated.csv
  generated_outputs:
    - smiles
    - pred_pIC50
    - target_pIC50
```

**Sampling modes explained:**

| Mode                     | Use Case                                       | Target Property                   |
| ------------------------ | ---------------------------------------------- | --------------------------------- |
| `single_target`        | Generate all molecules toward one fixed target | Fixed value (e.g., pIC50=8.0)     |
| `training_dist`        | Generate toward distribution of training data  | Gaussian(mean(train), std error)  |
| `uniform_range`        | Generate evenly across property range          | Uniform in [min, max]             |
| `uniform_range_strict` | Generate with quota per bin                    | Uniform with per-bin count target |

### 4. Analysis Configuration

```yaml
analysis:
  enabled: true
  profile: bace_pic50_10k
  overrides:
    save_distribution_plot: true
    save_scaffold_plot: false
    run_prediction_error_plot: true
    run_residual_plot: true
    run_chemical_space: true            # PCA/t-SNE of descriptor space
    debug: true
```

### 5. Cleanup & Output Management

```yaml
cleanup:
  remove_iteration_logs_after_success: false      # Keep .log files
  remove_iteration_data_after_success: true       # Delete temporary data/
  remove_sampling_debug_after_success: true       # Delete debug artifacts
  purge_python_caches_before_run: true            # Clean __pycache__
```

---

## Output Structure

### Per-Fold Iteration (`cv_iteration_k`)

```
fold_pipeline_outputs/cv_iteration_0/
├── generated/
│   ├── generated.csv                    <- Sampled SMILES + properties
│   └── quality_summary.csv              <- Stats (unique, valid, duplicates, etc.)
├── analysis/
│   ├── analysis_summary.json            <- Raw metrics & statistics
│   ├── scaffold_stats.json
│   ├── distribution_plot.png            <- Property distribution
│   ├── prediction_error_plot.png        <- Model error analysis
│   ├── residual_plot.png                <- Residuals vs. target
│   ├── tanimoto_histogram.png           <- Similarity to training set
│   ├── chemical_space_pca.png           <- PCA of molecular descriptors
│   └── chemical_space_tsne.png          <- t-SNE of molecular descriptors
└── logs/
    └── fold_pipeline_cv_iteration_0.log

save/fold_pipeline_runs/cv_iteration_0/training/
├── model_best.ckpt-63.pt                <- Best checkpoint
├── training_config.json                 <- Model + training config
├── history.csv                          <- Training loss history
└── train_validation_prediction_eval.csv <- Per-sample training predictions
```

### Cross-Fold Aggregation (`cv_combo`)

```
fold_pipeline_outputs/cv_combo/
├── cv_combo_metrics_stats.json          <- Aggregated metrics (mean/std across folds)
├── cv_combo_error_stats.csv             <- Error statistics across all folds
├── cv_combo_metrics_summary.png        <- Summary plot
└── cv_combo_metrics_boxplots.png        <- Per-metric boxplots across folds
```

### Global Summary

```
fold_pipeline_outputs/
└── cross_fold_analysis_summary.json     <- V/U/N aggregated stats per fold + combined + Diversity (external)
```

---

## Key Pipeline Concepts

### Sampling Quality Metrics (V, U, N)

After sampling, molecules are tracked by their **validity**:

- **V**: Total valid molecules (parseable by RDKit)
- **U**: Unique molecules (after removing duplicates)
- **N**: Novel molecules (not in training data)

These are aggregated per fold and across all folds in `cross_fold_analysis_summary.json`.

### Scaffold Filtering

The pipeline filters generated molecules by **Murcko scaffolds** to exclude:

1. **Training scaffolds** (if `exclude_training=true`): Exact scaffolds seen during training
2. **Validation scaffolds** (if `exclude_validation_scaffolds=true`): Scaffolds in the validation fold
3. **Heldout scaffolds** (if `exclude_heldout_scaffolds=true`): Scaffolds in a separate held-out test set

This ensures generated molecules are structurally *novel*.

### Label Head (Multi-task Learning)

When `predict_labels=true`, the model trains a separate **label prediction head** on top of the latent space:

- Input: latent vector **z** (optionally + condition **c** if `include_condition_in_label_head=true`)
- Output: predicted property values (e.g., pIC50)
- Loss: MSE with weight `label_loss_weight`
- Purpose: Force latent space to encode property information (improves sampling)

### Beta Annealing for Disentanglement

The **β-CVAE** optimization gradually increases β during training ([`start_beta` -&gt; `max_beta`](#2-training-configuration)):

- **Low β** (early epochs): Allow model to reconstruct with flexible latent space
- **High β** (later epochs): Constrain latent space toward standard normal prior (disentanglement)

This helps the label head and sampling work better by organizing the latent space.

---

## Advanced Usage

### Running with Different Random Seeds

To compare multiple sampling runs with different randomness:

```bash
# Fold 0, seed 42
python run_fold_pipeline.py --config fold_pipeline_config.example.yaml --only-fold 0 --seed 42

# Fold 0, seed 123
python run_fold_pipeline.py --config fold_pipeline_config.example.yaml --only-fold 0 --seed 123
```

### Using LSTM Model Instead of Transformer

Modify `fold_pipeline_config.example.yaml`:

```yaml
train:
  base_config:
    model:
      mode: lstm         # <- Change from transformer
```

LSTM is simpler but may be less expressive. Transformer is more powerful but requires careful tuning (dropout, weight decay) to avoid overfitting on small datasets.

### Customizing Post-Training Analysis

Modify `fold_pipeline_config.example.yaml`:

```yaml
analysis:
  overrides:
    save_distribution_plot: true
    run_chemical_space: true              # Enable expensive t-SNE/PCA
    save_scaffold_plot: false             # Disable if not needed
```

Analysis plots are saved under `fold_pipeline_outputs/cv_iteration_k/analysis/`.

---

## Performance Notes

### GPU Requirements

- Transformer model: ~6gb GPU VRAM for batch_size=64 with transformer, i used 3070
- LSTM: 4-6 GB GPU VRAM

### Sampling Speed

- Sampling 10k unique molecules: 10-20 minutes per fold (depends on generation efficiency)

### Analysis Speed

- Per-fold analysis: 2-5 minutes (depends on plot resolution and dataset size)
- Cross-fold aggregation: < 1 minute

---

## Notes

- **Configuration validation**: The pipeline validates YAML config at startup. Invalid keys raise errors.
- **Backward compatibility**: JSON configs still supported but YAML is preferred for readability.
- **Checkpoints**: PyTorch `.pt` format. Use `--load-checkpoint` to resume training.
- **Dependencies**: RDKit, PyTorch, PyYAML, Pandas, NumPy, Matplotlib, scikit-learn
- **Python version**: 3.9+
