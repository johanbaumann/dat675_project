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

The labels will be sampeled from latent space: $f(z)$. This head is separate from the decoder. The label head can be toggled to take the target into account. To be able to sucsessfully label the labels one needs an "disentangeled" latentspace. Disentangeled meaning that molecules sharing properties are in distinct "chunk like" areas of the latentspace. This is since an entangeled latentspace could mean that if one samples randomly, eventough the sample lies close to another from the training set, it does not nesecarily imply that they share features and labels (think of it like activitycliffs laying everywhere in propertyspace but this time in latentspace). And to actually disentangle latentspace one can use a larger  $\beta$ term. This is based on research done as mentioned by Guang Jun et al. The labels generated by the label head do have errors in pIC50, so they should be seen more as "fussy" than hard labels.

---

## BACE baseline workflow (pIC50-only conditioning)

This repo now supports a full BACE pipeline where the conditioning target is only `pIC50`.

### 1) Build BACE property file + descriptor stats

Run:

```bash
python cal_prop.py
```

With the default `args` in `cal_prop.py` (`mode='from_csv_columns'`), this will:

- read `bace.csv` (`mol` as SMILES, `pIC50` as target),
- write `bace_pic50.txt` in training format: `SMILES<TAB>pIC50`,
- write metadata sidecar `bace_pic50.txt.meta.json` with property names,
- extract descriptor inventory/stats from the original BACE CSV and write:
  - `bace_descriptor_stats.csv` (per-descriptor summary),
  - `bace_descriptor_summary.json` (descriptor list + dataset counts),
- print descriptor summary counts to console.

### 2) Train one CVAE for each of the 5 cv_iterations

`train_labels.py` defaults are set for a small-dataset baseline run:

- `data.prop_file='bace_pic50.txt'`
- `model.mode='lstm'`
- `training.batch_size=32`
- `training.num_epochs=120`
- `model.label_target_indices=None` (auto-selects valid target columns)

Run:

```bash
python train_labels.py
```

#### ONE STABLE config:

**Transformer**:

```

{
  "workspace_root": ".",
  "training_output_root": "save/fold_pipeline_runs",
  "artifacts_output_root": "fold_pipeline_outputs",
  "train_validation_folds_dir": "../Preprocessing/combination_1300_molecules_and_0_%_synthetic",
  "fold_glob": "fold_*.csv",
  "smiles_column": "smiles",
  "label_columns": [
    "pIC50"
  ],
  "python_executable": null,
  "train": {
    "enabled": false,
    "script": "train_labels.py",
    "base_config": {
      "training_preset": "custom",
      "data": {
        "seq_length": 120,
        "smiles_augmentation_duplicates": 10
      },
      "model": {
        "mode": "transformer",
        "latent_size": 200,
        "unit_size": 256, 
        "n_rnn_layer": 3,
        "mean": 0.0,
        "stddev": 1.0,
        "num_prop": null,
        "predict_labels": true,
        "label_target_indices": null,
        "label_dim": null,
        "label_loss_weight": 1.0,
        "include_condition_in_label_head": true,
        "label_targets_use_raw_scale": false
      },
      "transformer": {
        "heads": 8,
        "ff_size": 512,
        "dropout": 0.2
      },
      "optimization": {
        "optimizer": "adamw",
        "lr": 1e-3,
        "weight_decay": 0.001,
        "use_amp": false,
        "amp_dtype": "float16",
        "grad_clip_norm": 4.0
      },
      "training": {
        "batch_size": 64,
        "num_epochs": 120,
        "save_dir": "save/",
        "run_name": null,
        "use_run_subdir": true,
        "save_every": 30,
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 0.001,
        "early_stopping_restore_best": true
      },
      "scheduler": {
        "enabled": true,
        "factor": 0.5,
        "patience": 2,
        "threshold": 0.001,
        "min_lr": 1e-06
      },
      "kl": {
        "enabled": true,
        "start_beta": 1.0,
        "max_beta": 4.0,
        "hold_epochs": 0,
        "warmup_epochs": 8
      },
      "diagnostics": {
        "every": 1
      }
    }
  },
  "sampling": {
    "enabled": true,
    "checkpoint_glob": "model_best.ckpt-*.pt",
    "run_training_dist": true,
    "training_dist_std_scale": 1.0,
    "training_dist_clip_n_std": 2.5,
    "training_dist_seed": null,
    "target_prop_mode": "mean_test_labels",
    "target_prop": [
      8.0
    ],
    "num_unique": 10000,
    "max_batches": 5000,
    "do_sample": true,
    "temperature": 0.9,
    "top_k": 20,
    "exclude_training": true,
    "exclude_validation_scaffolds": true,
    "exclude_heldout_scaffolds": true,
    "heldout_smiles_csv": "../Preprocessing/heldout_datasets/heldout_testset.csv",
    "validation_smiles_column": "smiles",
    "heldout_smiles_column": "smiles",
    "scaffold_make_generic": false,
    "strip_salts": true,
    "decharge": true,
    "canonicalize_tautomer": true,
    "suppress_rdkit_parse_errors": true,
    "require_neutral": true,
    "mw_tolerance": null,
    "logp_tolerance": null,
    "min_tpsa": null,
    "max_heavy_atoms": 60,
    "max_canonical_smiles_len": null,
    "generated_outputs": [
      "smiles",
      "pred_pIC50",
      "target_pIC50"
    ],
    "save_generated_csv": true,
    "save_quality_summary": true,
    "result_filename": "generated.csv",
    "quality_summary_filename": "quality_summary.csv"
  },
  "analysis": {
    "enabled": true,
    "script": "run_viz_pipeline.py",
    "profile": "bace_pic50_10k",
    "overrides": {
      "save_distribution_plot": true,
      "save_scaffold_plot": true,
      "save_scaffold_grids": true,
      "run_prediction_error_plot": true,
      "run_train_loss_plot": true,
      "run_tanimoto_histogram": true,
      "run_chemical_space": true,
      "run_descriptor_space": false,
      "debug": true
    }
  },
  "cv_combo": {
    "enabled": true,
    "only": false,
    "cross_fold_summary_path": null
  }
}

```

### 3) Sample molecules

For pIC50-only runs, there is no direct RDKit pIC50 descriptor to enforce by tolerance at sampling time.

So in `sample_labels.py` defaults:

- `filters.mw_tolerance=None`
- `filters.logp_tolerance=None`

Sampling therefore accepts valid/novel molecules after canonicalization/cleanup filters, without target-proximity rejection based on MW/LogP.

Run:

```bash
python sample_labels.py
```

The generated table still includes RDKit-computed `MW`, `LogP`, `TPSA` for inspection, plus any predicted label columns from the label head.

1) Prepare SMILES property file

Input: one SMILES per line in `smiles.txt`.

```bash
python cal_prop.py
```

Set input/output filenames in the `args` dict at the top of `cal_prop.py`.

### Sampling with predicted labels (`sample_labels.py`)

If you trained a checkpoint with the optional label head (`train_labels.py` / `models/model_labels.py`), you can sample with `sample_labels.py`.

It writes the usual RDKit descriptors (MW/LogP/TPSA) plus:

- `pred_label_<j>`: raw head outputs per dimension.
- `pred_<name>`: denormalized predictions when the checkpoint contains label metadata (example: `pred_LogP`).

### Sampling target properties near training data (`training_dist` mode)

`sample_labels.py` supports a 3rd conditioning mode besides single-target and sweep:

- **Single target:** fixed `target_prop` for all samples.
- **Sweep:** `sweep.enabled=True` with a grid over target properties.
- **Training distribution:** `training_dist.enabled=True` to sample target properties around the training-data distribution.

This is useful when you care about the label head being accurate around *typical* training values, not just at a hand-picked target.

It uses the training-property mean/std saved in the checkpoint config (`prop_norm_mean/std`) and samples in raw units:

$$
c_{raw} \sim \mathcal{N}(\mu, (\text{std\_scale} \cdot \sigma)^2)
$$

Then it applies the same normalization as training before feeding `c` to the model.

Knobs in the `runtime_config` block inside `sample_labels.py`:

- `training_dist.enabled`: enable this mode.
- `training_dist.std_scale`: widen/narrow the sampled distribution (1.0 = match training std).
- `training_dist.clip_n_std`: clip each dimension to `mean +/- clip_n_std * std` to avoid extreme tails.
- `training_dist.seed`: optional reproducible seed.

Note: in training-dist mode the per-sample targets vary, so the fixed-target acceptance predicate (MW/LogP tolerance around a single `target_prop`) is disabled by default.

### Sampling controls (diversity vs validity)

This repo uses token-by-token decoding for SMILES generation. If you decode with greedy `argmax` at every step, the model can *collapse* and output the same molecule repeatedly (even with different latent vectors). To avoid this, `model.sample()` uses supports top-k temperature controled stochastic decoding and `sample_labels.py` has:

- `do_sample`: if `True`, samples the next token from the model distribution; if `False`, uses greedy decoding.
- `temperature`: scales logits before sampling. Lower values generally improve validity but reduce diversity.
- `top_k`: restricts sampling to the `k` most probable tokens per step (often improves validity).

The defaults in `sample_labels.py` are set for faster unique generation (tune as needed for your checkpoint/dataset).

### Training-set novelty filter cache

When `exclude_training=True`, `sample_labels.py` loads canonical SMILES from the training/property file so it can reject molecules already seen during training. For large files, canonicalization is expensive, so the loader now writes a cache next to the property file:

- `prop_mw_logp.txt.canon_seq120.pkl.gz`

The cache is best-effort and automatically invalidated if the property file is newer or the `seq_length` changes.

## Numerical stability notes

## Module-based analysis pipeline (keeps `viz.ipynb` intact)

If you want the same analysis flow as `viz.ipynb` but from reusable Python modules,
use the module runner:

```bash
python run_viz_pipeline.py --config analysis_run_config.json
```

The analysis config is file-driven (JSON), so you do **not** need to pass many CLI
arguments anymore.

The checked-in `analysis_run_config.json` is runnable against the included fold outputs. You can still override `overrides.train_folder`, `overrides.generated_data_path`, and `overrides.output_dir` for your own runs.

### Starter config file

Use and edit:

```text
analysis_run_config.json
```

Key fields to edit:

- `profile`: `bace_pic50_10k`
- `overrides.train_folder`
- `overrides.train_data_path`
- `overrides.generated_data_path`
- `overrides.output_dir`
- `overrides.target_property_column`
- `overrides.predicted_property_column`

### What the module pipeline now covers

The pipeline in `analysis_modules/` includes the main parts of the notebook workflow:

- training history plot (`train_loss` / `test_loss`) from `train_folder/history.csv`
- Tanimoto similarity to training set + diversity score
- Tanimoto distribution histogram
- prediction-error scatter (`abs error vs ground truth`) + summary metrics (MSE/MAE/median/std)
- train/generated property distributions (y-axis is `Count`; generated uses predicted property when available)
- MW train-vs-generated difference plot (`train - generated` counts per bin)
- chemical-space PCA (Morgan fp) and chemical-space t-SNE
- descriptor-space PCA and descriptor-space t-SNE
- generated-space coloring by max Tanimoto where applicable
- scaffold overlap/distribution summary
- top-scaffold grid images for train and generated sets (+ optional novel generated scaffolds grid)
- explicit scaffold stats artifact with unique scaffold counts (train vs generated, overlap, novel)

It writes processed CSV + analysis summary JSON + figures under `output_dir`.

### BACE pIC50 + 10k generated molecules

`analysis_run_config.json` is written for the active BACE pIC50 workflow.

- Transformer encoder/decoder blocks run in fp32 under AMP (`selective autocast`) to avoid fp16 softmax/masked-attention NaNs.
- Reconstruction loss is length-masked (padded tokens do not contribute to CE).
- If you see instability/NaNs with Transformer, try disabling AMP (`use_amp=False`) until the model is stable.
- KL annealing defaults are set to safer values for early training: `start_beta=0.01`, `hold_epochs=0`, `warmup_epochs=50`.

## Transformer implementation notes

### Dataset padding + length convention

`utils.load_data()` constructs training pairs using:

- input sequence: `'X' + smiles`, padded with `'E'` to `seq_length`
- target sequence: `smiles`, padded with `'E'` to `seq_length`
- length: `len(smiles) + 1` (includes the leading `'X'` step)

This means the model learns to predict the first SMILES token given `'X'`, and also learns to predict the terminal `'E'` right after the last SMILES character.

### Why `tgt_key_padding_mask` is required (Transformer)

Even if the reconstruction loss ignores padded positions, a Transformer decoder can still attend to padded `'E'` tokens in the target sequence during self-attention unless `tgt_key_padding_mask` is provided. That leakage can cause degenerate behavior (e.g., predicting only `'E'`), especially early in training.

This repo passes `tgt_key_padding_mask` in the Transformer decoder when `lengths` are available (training). During autoregressive sampling, `lengths` is not known ahead of time and the mask is omitted.

### Reconstruction loss masking

Reconstruction loss is computed token-wise and masked by `lengths` so that positions `>= length` do not contribute.

## Known failure modes

- Symptom: model predicts only padding (`'E'`) for most positions
  - Cause: missing `tgt_key_padding_mask` and/or reconstruction loss not masked by sequence length
- Symptom: unstable training / NaN loss (Transformer)
  - Cause: AMP fp16 overflow in attention/softmax or extreme latent log-variance; try `use_amp=False` and/or check `log_sigma` clamp

## Architecture overview (Transformer mode)

- Encoder
  - Token embedding (`latent_size`)
  - Concatenate conditioning properties per time step
  - Linear projection to `d_model=unit_size` + positional encoding
  - `TransformerEncoder` with `src_key_padding_mask`
  - Pool last valid state (by `lengths`) to latent mean/log-variance
- Decoder
  - Token embedding (`latent_size`)
  - Concatenate latent `z` and conditioning properties per time step
  - Linear projection to `d_model=unit_size` + positional encoding
  - `TransformerDecoder` with causal mask + `tgt_key_padding_mask`
  - Linear projection to vocabulary logits

## Notes

- This codebase uses PyTorch checkpoints (`.pt`).
- Transformer path is implemented to avoid warning-prone mask/load patterns in recent PyTorch versions.
