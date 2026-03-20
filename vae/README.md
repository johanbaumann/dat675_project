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

This repo now runs a single top-level CV pipeline with one main config file.

### Single entrypoint + single config

- Entrypoint: `run_fold_pipeline.py`
- Config file: `fold_pipeline_config.example.json`

Run full CV pipeline from workspace root:

```powershell
python run_fold_pipeline.py --config fold_pipeline_config.example.json
```

Run one CV iteration only:

```powershell
python run_fold_pipeline.py --config fold_pipeline_config.example.json --only-fold 0
```

### What the pipeline does per CV iteration

1. Use one fold CSV as validation.
2. Merge remaining fold CSVs as training data.
3. Train one model (`scripts/train_labels.py`) if `train.enabled=true`.
4. Sample generated molecules (`pipeline/sampling_pipeline.py`) if `sampling.enabled=true`.
5. Run analysis in-process from `run_fold_pipeline.py` if `analysis.enabled=true`.

This means training, sampling, and analysis are controlled from one script and one config.

### Required/important config keys

- `train_validation_folds_dir`
- `fold_glob`
- `smiles_column`
- `label_columns`
- `training_output_root`
- `artifacts_output_root`
- `train.base_config`
- `sampling.*`
- `analysis.*`
- `cleanup.*`
- `cv_combo.*`

### Analysis-only mode (no re-training, no re-sampling)

Set:

- `train.enabled=false`
- `sampling.enabled=false`
- `analysis.enabled=true`

In this mode, the pipeline reuses:

- `artifacts_output_root/cv_iteration_<k>/generated/generated.csv`
- `artifacts_output_root/cv_iteration_<k>/generated/quality_summary.csv`

Behavior:

- Per-fold analysis still runs for each detected CV iteration.
- Missing/empty `generated.csv` hard-fails that iteration.
- Missing `quality_summary.csv` allows analysis to continue, but cross-fold V.U.N aggregation for that fold is skipped.

### CV combo-only mode

Set:

- `cv_combo.enabled=true`
- `cv_combo.only=true`

Optional:

- `cv_combo.cross_fold_summary_path` to use a specific summary JSON.

Outputs under `artifacts_output_root/cv_combo/`:

- `cv_combo_metrics_summary.png`
- `cv_combo_metrics_boxplots.png`
- `cv_combo_metrics_stats.json`
- `cv_combo_error_stats.csv`

### Sampling target modes

Use `sampling.target_sampling_mode`:

- `single_target`
- `training_dist`
- `uniform_range`
- `uniform_range_strict`

For pIC50-only conditioning (`num_prop=1`), use one-dimensional targets/ranges.

### Heldout and scaffold exclusion

Heldout data is used for scaffold filtering (not training):

- `sampling.exclude_training`
- `sampling.exclude_validation_scaffolds`
- `sampling.exclude_heldout_scaffolds`
- `sampling.heldout_smiles_csv`
- `sampling.validation_smiles_column`
- `sampling.heldout_smiles_column`
- `sampling.scaffold_make_generic`

### Output layout

Per iteration under `artifacts_output_root/cv_iteration_<k>/`:

- `data/`
- `generated/`
- `analysis/`
- `logs/`
- `iteration_manifest.json` (if enabled)

Training artifacts under `training_output_root/cv_iteration_<k>/training/`.

Global outputs:

- `artifacts_output_root/global_manifest.json` (if enabled)
- `artifacts_output_root/cross_fold_analysis_summary.json`

### Numerical stability notes

- Transformer attention blocks can be sensitive with AMP in some settings.
- Reconstruction loss is length-masked so padded tokens do not contribute.
- If training gets unstable, set `optimization.use_amp=false` and re-test.

### Notes

- This codebase uses PyTorch checkpoints (`.pt`).
- Keep using `run_fold_pipeline.py` + `fold_pipeline_config.example.json` as the primary interface.
