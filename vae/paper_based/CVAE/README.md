![Screenshot](figure.png)

# Conditional VAE for molecular generation (PyTorch)

Reference paper:

- https://jcheminf.biomedcentral.com/articles/10.1186/s13321-018-0286-7
- https://arxiv.org/abs/1806.05805

This repository now contains an extended implementation that supports both:

- `lstm` CVAE (paper-style baseline), and
- `transformer` CVAE (your extension).

## What is different from the original paper implementation

Your modifications in this repo include:

- Dual architecture switch in one `CVAE` class: `model_mode = lstm | transformer`.
- Saved training/model recreation config (`training_config.json`) during training.
- Sampling that can auto-load training config from the checkpoint folder (no manual architecture retyping).
- Modular config helpers in `utils.py` for defaults, JSON load/save, and compose-from-overrides.
- Improved generation filtering/reporting in sampling (`unique`, `invalid`, `duplicates`, `in_training`).
- EOS-aware early stopping in decode loop for faster generation.

## 1) Prepare SMILES property file

Input: one SMILES per line in `smiles.txt`.

```bash
python cal_prop.py
```

Set input/output filenames in the `args` dict at the top of `cal_prop.py`.

`cal_prop.py` is now modular via a descriptor registry. Configure properties in the in-file `args` dict:

```python
"properties": ['MW', 'LogP']
```

Supported descriptor names:

- `MW`
- `LogP`
- `TPSA`
- `NumHBD`
- `NumHBA`

You can use any subset/order. The selected order becomes the conditioning-column order in `smiles_prop.txt` and must match `sample.py` `target_prop` order.

## 2) Train model

Training is configured directly inside `train.py` using a single `config` dictionary.

### Configure in `train.py`

Edit the grouped `config` block near the top of `train.py`.

```python
config = {
	'training_preset': 'custom',
	'data': {'prop_file': 'prop_mw_logp.txt', 'seq_length': 120, 'train_ratio': 0.75},
	'model': {'mode': 'transformer', 'latent_size': 200, 'unit_size': 512, 'n_rnn_layer': 2},
	'transformer': {'heads': 8, 'ff_size': 1024, 'dropout': 0.15},
	'optimization': {'optimizer': 'adamw', 'lr': 1e-4, 'use_amp': True, 'amp_dtype': 'float16'},
	'training': {'batch_size': 64, 'num_epochs': 100, 'save_dir': 'save/'},
	'scheduler': {'enabled': True, 'factor': 0.5, 'patience': 5, 'threshold': 1e-4, 'min_lr': 1e-6},
	'kl': {'enabled': True, 'start_beta': 0.01, 'max_beta': 1.0, 'hold_epochs': 0, 'warmup_epochs': 50},
	'diagnostics': {'every': 1},
}
```

Grouped config is flattened internally, so legacy flat keys are still supported for compatibility.

In Transformer mode, embedding width is still `latent_size`, while internal attention/FFN width is `unit_size`.

### Run training

```bash
python -u train.py
```

No external config file is required to start training.

`train.py` now infers `num_prop` directly from `smiles_prop.txt` (number of numeric columns after SMILES), so you do not need to hardcode it.

### Training outputs

Each run saves:

- checkpoint: `save_dir/model_.ckpt-<epoch>.pt`
- training history: `save_dir/history.csv`
- model recreation config: `save_dir/training_config.json`

## 3) Generate molecules (sampling)

`sample.py` now uses an internal config block. Set:

- `save_file` to a trained checkpoint path,
- `target_prop` to desired property values in the same order as `cal_prop.py` `args["properties"]`.

By default, it auto-loads `training_config.json` from the same folder as `save_file`.

For MW + LogP training, use:

```python
'target_prop': '300.0 3.0'
```

`sample.py` validates that the number of values in `target_prop` matches the trained model/property-file dimensionality.

Example:

```bash
python sample.py
```

Output is written to `result_filename` (default `result.txt`).

### Sampling controls (diversity vs validity)

This repo uses token-by-token decoding for SMILES generation. If you decode with greedy `argmax` at every step, the model can *collapse* and output the same molecule repeatedly (even with different latent vectors). To avoid this, `model.sample()` now supports stochastic decoding and `sample.py` exposes these knobs:

- `do_sample`: if `True`, samples the next token from the model distribution; if `False`, uses greedy decoding.
- `temperature`: scales logits before sampling. Lower values generally improve validity but reduce diversity.
- `top_k`: restricts sampling to the `k` most probable tokens per step (often improves validity).

The defaults in `sample.py` are set for faster unique generation (tune as needed for your checkpoint/dataset).

### Training-set novelty filter cache

When `exclude_training=True`, `sample.py` loads canonical SMILES from the training/property file so it can reject molecules already seen during training. For large files, canonicalization is expensive, so the loader now writes a cache next to the property file:

- `prop_mw_logp.txt.canon_seq120.pkl.gz`

The cache is best-effort and automatically invalidated if the property file is newer or the `seq_length` changes.

### Sweep results (this checkpoint)

To quantify which settings work well for a given checkpoint, run:

```bash
python sweep_sampling.py
```

It runs a small grid over `temperature` and `top_k` and reports the fraction of samples that become **unique + novel + RDKit-valid** after filtering (plus invalid/duplicate rates). For `save/model_9.ckpt-9.pt` with 25 batches (1600 samples) per setting and target properties MW=300, LogP=3, the sweep produced:

| temperature | top_k | samples | accepted (unique+novel) | accepted_rate | invalid_rate | dup_rate | in_training_rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.6 | 20 | 1600 | 526 | 32.88% | 56.88% | 10.25% | 0.00% |
| 0.6 | 100 | 1600 | 521 | 32.56% | 58.06% | 9.38% | 0.00% |
| 0.6 | 50 | 1600 | 518 | 32.38% | 57.56% | 10.06% | 0.00% |
| 0.6 | 10 | 1600 | 501 | 31.31% | 59.44% | 9.25% | 0.00% |
| 0.7 | 100 | 1600 | 289 | 18.06% | 81.44% | 0.50% | 0.00% |
| 0.7 | 20 | 1600 | 275 | 17.19% | 82.31% | 0.50% | 0.00% |
| 0.7 | 50 | 1600 | 270 | 16.88% | 83.06% | 0.06% | 0.00% |
| 0.7 | 10 | 1600 | 267 | 16.69% | 83.00% | 0.31% | 0.00% |
| 0.8 | 50 | 1600 | 108 | 6.75% | 93.25% | 0.00% | 0.00% |
| 0.8 | 20 | 1600 | 104 | 6.50% | 93.44% | 0.06% | 0.00% |
| 0.8 | 10 | 1600 | 102 | 6.38% | 93.56% | 0.06% | 0.00% |
| 0.8 | 100 | 1600 | 95 | 5.94% | 93.94% | 0.12% | 0.00% |
| 0.9 | 100 | 1600 | 51 | 3.19% | 96.69% | 0.12% | 0.00% |
| 0.9 | 50 | 1600 | 51 | 3.19% | 96.75% | 0.06% | 0.00% |
| 0.9 | 10 | 1600 | 43 | 2.69% | 97.19% | 0.12% | 0.00% |
| 0.9 | 20 | 1600 | 42 | 2.62% | 97.31% | 0.06% | 0.00% |
| 1.0 | 10 | 1600 | 38 | 2.38% | 97.31% | 0.31% | 0.00% |
| 1.0 | 20 | 1600 | 36 | 2.25% | 97.62% | 0.12% | 0.00% |
| 1.0 | 50 | 1600 | 33 | 2.06% | 97.81% | 0.12% | 0.00% |
| 1.0 | 100 | 1600 | 20 | 1.25% | 98.44% | 0.31% | 0.00% |

Interpretation:

- For this checkpoint, `temperature=0.6` with `top_k` in the 10–100 range produced the highest unique+novel acceptance rate.
- Higher temperatures greatly increased invalid SMILES rate for this model.

## Numerical stability notes

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

This repo now passes `tgt_key_padding_mask` in the Transformer decoder when `lengths` are available (training). During autoregressive sampling, `lengths` is not known ahead of time and the mask is omitted.

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
