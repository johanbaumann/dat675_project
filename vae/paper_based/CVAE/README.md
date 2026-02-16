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

Edit the `config` block near the top of `train.py`.

For LSTM mode:

```python
'model_mode': 'lstm'
```

For Transformer mode:

```python
'model_mode': 'transformer',
'transformer_heads': 8,
'transformer_ff_size': 2048,
'transformer_dropout': 0.1,
```

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

## Notes

- This codebase uses PyTorch checkpoints (`.pt`).
- Transformer path is implemented to avoid warning-prone mask/load patterns in recent PyTorch versions.
