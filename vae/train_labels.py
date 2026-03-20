"""train_label.py

Train script for CVAE with property conditioning.

This script now optionally trains an auxiliary label predictor head (see
`models/model_labels.py`) that predicts molecular labels from the latent vector `z`.

Important invariants (kept):
- `load_data()` signature and output semantics are unchanged.
- Conditioning vector `c` is unchanged (still taken from the property file).
- Sampling pipeline is unchanged.

CHANGELOG
---------
2026-02-23
- Added optional label predictor training via `predict_labels` config knobs.
- Uses `models.model_labels.CVAE` so the new head can be trained without changing the
    base `model.py` implementation.
"""

from models.model_labels import CVAE
from utils.core import (
    build_train_run_save_dir,
    compose_train_config_from_dict,
    convert_to_smiles,
    ensure_dir,
    get_model_config,
    load_condition_property_names,
    load_data,
    save_training_config,
    split_train_test,
)
from utils.labels import (
    apply_training_preset,
    get_kl_beta,
    log_cuda_mem,
    save_best_checkpoint,
    save_current_checkpoint,
    save_history_csv,
)
import argparse
import json
import os
from typing import Optional
import numpy as np
import time
import pandas as pd
import torch
from rdkit import Chem
from copy import deepcopy


def _deep_update_dict(base: dict, override: dict) -> dict:
    """Recursively update nested dict values while keeping unspecified defaults."""
    out = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update_dict(out[key], value)
        else:
            out[key] = value
    return out


def _parse_config_override_path_from_argv() -> str:
    """Read required config override path from CLI.

    In pipeline-only mode, `train_labels.py` must be invoked by
    `run_fold_pipeline.py` with a generated per-iteration JSON config.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config-json', type=str, required=True)
    args, _ = parser.parse_known_args()
    return str(args.config_json)


def _load_runtime_config_override() -> dict:
    """Load required JSON config override for fold-pipeline orchestration."""
    cfg_path = _parse_config_override_path_from_argv()
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f'--config-json path does not exist: {cfg_path}')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError('Config override JSON must be an object (dict).')
    print(f'loaded runtime config override from: {cfg_path}')
    return payload

## Single source of truth for run configuration.
## Grouped sections are easier to edit; utils will flatten this to legacy keys.
#config = {
#    'training_preset': 'custom',  # 'custom' or 'stable_transformer'
#    'data': {
#        'prop_file': 'bace_pic50.txt',
#        'seq_length': 120,
#        'train_ratio': 0.8,
#        # Number of randomized SMILES to add per original TRAIN sample.
#        # 0 disables augmentation.
#        'smiles_augmentation_duplicates': 0,
#    },
#    'model': {
#        'mode': 'transformer',  # 'lstm' or 'transformer'
#        'latent_size': 200,
#        'unit_size': 512,
#        'n_rnn_layer': 2, # 2 layers for transformers, 2-3 for lstm
#        'mean': 0.0,
#        'stddev': 1.0,
#        'num_prop': None,  # inferred from property file
#
#        # Optional multi-task head:
#        # - predict_labels=False keeps the model identical to the base CVAE.
#        # - When enabled, the model predicts a label vector from latent `z`.
#        'predict_labels': True,
#        # Which conditioning columns to predict as labels.
#        # Example (for: MW, LogP setup): set to [1] to predict LogP only.
#        # If None and predict_labels=True, defaults to predicting all properties.
#        'label_target_indices': None,
#        'label_dim': None,  # if None, inferred from property file
#        'label_loss_weight': 1.0, # relative weight of label prediction loss compared to reconstruction + KL loss (which are weighted by beta)
#
#        # Optional label-head variants:
#        # If True, the label head predicts from both (z, c) instead of z only.
#        # This can be useful if you want predicted labels to track the sampling
#        # target properties more directly.
#        'include_condition_in_label_head': False,
#
#        # If True, train the label head on *raw* (unnormalized) property values.
#        # If False (default), train it on the normalized conditioning vector.
#        'label_targets_use_raw_scale': False,
#    },
#    'transformer': {
#        'heads': 8,
#        'ff_size': 1024,
#        'dropout': 0.15,
#    },
#    'optimization': {
#        'optimizer': 'adamw', # 'adam' for lstm, 'adamw' for transformer
#        'lr': 1e-5,
#        'weight_decay': 0.001, # 0.001 for transformer
#        'use_amp': False,
#        'amp_dtype': 'float16',
#        'grad_clip_norm': 4.0,
#    },
#    'training': {
#        'batch_size': 64,
#        'num_epochs': 120,
#        'save_dir': 'save/',
#        'run_name': None,  # If None, auto-generated timestamped run folder is used.
#        'use_run_subdir': True,  # If True, save into save_dir/<run_name_or_timestamp>/
#        'save_every': 10,
#        'early_stopping_patience': 10,
#        'early_stopping_min_delta': 0.001,
#        'early_stopping_restore_best': True,
#    },
#    'scheduler': {
#        'enabled': True,
#        'factor': 0.5,
#        'patience': 2,
#        'threshold': 1e-3, # minimum change in the monitored quantity to qualify as an improvement (for 'min' mode).
#        'min_lr': 1e-6,
#    },
#    'kl': {
#        'enabled': True,
#        'start_beta': 1.0, # start with low KL weight to allow model to learn reconstruction before regularizing latent space, can help with stability (especially for transformer + amp).
#        'max_beta': 4.0,
#        'hold_epochs': 0,
#        'warmup_epochs': 8,
#    },
#    'diagnostics': {
#        'every': 1,
#    },
#}

config = {
    'training_preset': 'custom',
    'data': {
        'prop_file': 'bace_pic50.txt',
        'seq_length': 120,
        'train_ratio': 0.8,
        # Number of randomized SMILES strings generated per original train sample.
        # 0 disables augmentation.
        'smiles_augmentation_duplicates': 10,
    },
    'model': {
        'mode': 'transformer',
        'latent_size': 128,
        'unit_size': 256,
        'n_rnn_layer': 2,
        'mean': 0.0,
        'stddev': 1.0,
        'num_prop': None,
        'predict_labels': True,
        'label_target_indices': None,
        'label_dim': None,
        'label_loss_weight': 1.0,
        'include_condition_in_label_head': False,
        'label_targets_use_raw_scale': False,
    },
    'transformer': {
        'heads': 4,
        'ff_size': 512,
        'dropout': 0.1,
    },
    'optimization': {
        'optimizer': 'adamw',
        'lr': 5e-5,
        'weight_decay': 0.001,
        'use_amp': False,
        'amp_dtype': 'float16', # use bfloat16 for transformer + amp.
        'grad_clip_norm': 2.0, # dont be too harh with clipping, can hinder learning. 
    },
    'training': {
        'batch_size': 64,
        'num_epochs': 120,
        'save_dir': 'save/',
        'run_name': None,
        'use_run_subdir': True,
        'save_every': 10,
        'early_stopping_patience': 10,
        'early_stopping_min_delta': 0.001,
        'early_stopping_restore_best': True,
    },
    'scheduler': {
        'enabled': True,
        'factor': 0.5, # reduce LR by this factor when test loss plateaus.
        'patience': 2, # number of epochs with no improvement after which LR will be reduced.
        'threshold': 1e-3, # minimum change in the monitored quantity to qualify as an improvement (for 'min' mode).
        'min_lr': 1e-6, # lower bound on the learning rate after reductions.
    },
    'kl': {
        'enabled': True,
        'start_beta': 1.0,
        'max_beta': 4.0,
        'hold_epochs': 0,
        'warmup_epochs': 8,
    },
    'diagnostics': {
        'every': 1,
    },
}

runtime_config_override = _load_runtime_config_override()
config = _deep_update_dict(config, runtime_config_override)
print('applied runtime config from --config-json on top of in-file baseline config')


def _augment_train_split_with_random_smiles(
    *,
    train_input: np.ndarray,
    train_output: np.ndarray,
    train_labels: np.ndarray,
    train_length: np.ndarray,
    charset,
    vocab: dict,
    seq_length: int,
    duplicates_per_smiles: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Augment TRAIN split by adding randomized SMILES variants.

    This keeps the original train/test split intact to avoid leakage from
    augmenting before splitting.
    """
    dups = int(duplicates_per_smiles)
    if dups <= 0:
        return train_input, train_output, train_labels, train_length, {
            'aug_added': 0,
            'skipped_invalid': 0,
            'skipped_too_long': 0,
            'skipped_unknown_token': 0,
        }

    charset_arr = np.asarray(charset)
    aug_input = []
    aug_output = []
    aug_labels = []
    aug_length = []

    skipped_invalid = 0
    skipped_too_long = 0
    skipped_unknown_token = 0

    for row_idx in range(int(train_output.shape[0])):
        base = convert_to_smiles(train_output[row_idx], charset_arr)
        base = base.split('E', 1)[0].strip()
        if not base:
            skipped_invalid += dups
            continue

        mol = Chem.MolFromSmiles(base)
        if mol is None:
            skipped_invalid += dups
            continue

        for _ in range(dups):
            aug_smi = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
            if not aug_smi:
                skipped_invalid += 1
                continue
            if len(aug_smi) >= int(seq_length) - 2:
                skipped_too_long += 1
                continue
            if any(ch not in vocab for ch in aug_smi):
                skipped_unknown_token += 1
                continue

            x_str = ('X' + aug_smi).ljust(int(seq_length), 'E')
            y_str = aug_smi.ljust(int(seq_length), 'E')

            aug_input.append(np.asarray(list(map(vocab.get, x_str)), dtype=np.int64))
            aug_output.append(np.asarray(list(map(vocab.get, y_str)), dtype=np.int64))
            aug_labels.append(np.asarray(train_labels[row_idx], dtype=np.float32))
            aug_length.append(int(len(aug_smi) + 1))

    if len(aug_input) == 0:
        return train_input, train_output, train_labels, train_length, {
            'aug_added': 0,
            'skipped_invalid': int(skipped_invalid),
            'skipped_too_long': int(skipped_too_long),
            'skipped_unknown_token': int(skipped_unknown_token),
        }

    train_input_aug = np.concatenate([train_input, np.stack(aug_input, axis=0)], axis=0)
    train_output_aug = np.concatenate([train_output, np.stack(aug_output, axis=0)], axis=0)
    train_labels_aug = np.concatenate([train_labels, np.stack(aug_labels, axis=0)], axis=0)
    train_length_aug = np.concatenate([train_length, np.asarray(aug_length, dtype=train_length.dtype)], axis=0)

    return train_input_aug, train_output_aug, train_labels_aug, train_length_aug, {
        'aug_added': int(len(aug_input)),
        'skipped_invalid': int(skipped_invalid),
        'skipped_too_long': int(skipped_too_long),
        'skipped_unknown_token': int(skipped_unknown_token),
    }


def _load_data_with_existing_vocab(prop_file: str, seq_length: int, vocab: dict) -> tuple:
    """Load a dataset split using an already-fixed vocabulary.

    This keeps train/test tokenization aligned when train and test are provided as
    separate files (e.g. scaffold split folds).
    """
    with open(prop_file, 'r', encoding='utf-8', errors='ignore') as f:
        raw_lines = f.read().split('\n')

    x_rows = []
    y_rows = []
    labels = []
    lengths = []
    skipped_too_long = 0
    skipped_empty = 0
    skipped_unknown_token = 0

    for raw in raw_lines:
        line = str(raw).strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            skipped_empty += 1
            continue

        smi = str(parts[0]).strip()
        if len(smi) >= int(seq_length) - 2:
            skipped_too_long += 1
            continue
        if any(ch not in vocab for ch in smi):
            skipped_unknown_token += 1
            continue

        try:
            prop_vals = [float(v) for v in parts[1:]]
        except Exception:
            skipped_empty += 1
            continue

        x_str = ('X' + smi).ljust(int(seq_length), 'E')
        y_str = smi.ljust(int(seq_length), 'E')

        x_rows.append(np.asarray([vocab[ch] for ch in x_str], dtype=np.int64))
        y_rows.append(np.asarray([vocab[ch] for ch in y_str], dtype=np.int64))
        labels.append(np.asarray(prop_vals, dtype=np.float32))
        lengths.append(int(len(smi) + 1))

    if len(x_rows) == 0:
        raise ValueError(
            f'No usable rows found in external split file {prop_file}. '
            f'skipped_empty={skipped_empty}, skipped_too_long={skipped_too_long}, '
            f'skipped_unknown_token={skipped_unknown_token}'
        )

    label_arr = np.stack(labels, axis=0).astype(np.float32)
    print(
        f'external split load: {prop_file}, rows={len(x_rows)}, '
        f'skipped_empty={skipped_empty}, skipped_too_long={skipped_too_long}, '
        f'skipped_unknown_token={skipped_unknown_token}'
    )

    return (
        np.stack(x_rows, axis=0),
        np.stack(y_rows, axis=0),
        label_arr,
        np.asarray(lengths, dtype=np.int64),
    )


def _decode_smiles_batch(token_rows: np.ndarray, charset: np.ndarray) -> list[str]:
    smiles_list: list[str] = []
    for row in token_rows:
        smi = convert_to_smiles(row, charset)
        smi = str(smi).split('E', 1)[0].strip()
        smiles_list.append(smi)
    return smiles_list


def _predict_label_outputs_for_split(
    *,
    model: CVAE,
    x_tokens: np.ndarray,
    cond_norm: np.ndarray,
    lengths: np.ndarray,
    seq_output_tokens: np.ndarray,
    labels_raw: np.ndarray,
    split_name: str,
    charset: np.ndarray,
    label_target_indices: list[int],
    label_target_names: list[str],
    label_target_scale_raw: bool,
    prop_norm_mean: np.ndarray,
    prop_norm_std: np.ndarray,
    batch_size: int,
) -> pd.DataFrame:
    if x_tokens.shape[0] == 0:
        return pd.DataFrame(columns=['split', 'smiles'])

    smiles = _decode_smiles_batch(seq_output_tokens, charset)
    pred_chunks: list[np.ndarray] = []
    model.train(False)

    with torch.no_grad():
        for start in range(0, int(x_tokens.shape[0]), int(batch_size)):
            end = min(int(x_tokens.shape[0]), start + int(batch_size))
            x_b = torch.as_tensor(x_tokens[start:end], dtype=torch.long, device=model.device)
            c_b = torch.as_tensor(cond_norm[start:end], dtype=torch.float32, device=model.device)
            l_b = torch.as_tensor(lengths[start:end], dtype=torch.long, device=model.device)

            # Use latent mean for deterministic label prediction export.
            _z_b, mean_b, _log_sigma_b = model.encode(x_b, c_b, l_b)
            y_hat_b = model.predict_label(mean_b, c=c_b)
            pred_chunks.append(y_hat_b.detach().cpu().numpy().astype(np.float32))

    pred_mat = np.concatenate(pred_chunks, axis=0).astype(np.float32)
    if pred_mat.ndim == 1:
        pred_mat = pred_mat.reshape(-1, 1)

    if bool(label_target_scale_raw):
        pred_raw = pred_mat
    else:
        mean_sel = prop_norm_mean[label_target_indices].astype(np.float32)
        std_sel = prop_norm_std[label_target_indices].astype(np.float32)
        std_sel = np.where(std_sel < 1e-8, 1.0, std_sel)
        pred_raw = (pred_mat * std_sel.reshape(1, -1)) + mean_sel.reshape(1, -1)

    out = pd.DataFrame({'split': [str(split_name)] * int(x_tokens.shape[0]), 'smiles': smiles})
    for j, idx in enumerate(label_target_indices):
        name = str(label_target_names[j]) if j < len(label_target_names) else f'prop_{idx}'
        out[str(name)] = labels_raw[:, idx].astype(np.float32)
        out[f'pred_{name}'] = pred_raw[:, j].astype(np.float32)
    return out


def _export_train_validation_prediction_eval_csv(
    *,
    model: CVAE,
    config: dict,
    charset: np.ndarray,
    train_molecules_input: np.ndarray,
    train_molecules_output: np.ndarray,
    train_length: np.ndarray,
    train_labels: np.ndarray,
    train_labels_raw: np.ndarray,
    test_molecules_input: np.ndarray,
    test_molecules_output: np.ndarray,
    test_length: np.ndarray,
    test_labels: np.ndarray,
    test_labels_raw: np.ndarray,
    label_target_indices: Optional[list[int]],
    label_target_names: Optional[list[str]],
    label_targets_use_raw_scale: bool,
    prop_mean: np.ndarray,
    prop_std: np.ndarray,
) -> Optional[str]:
    if not bool(config.get('save_prediction_eval_csv', True)):
        return None
    if label_target_indices is None or len(label_target_indices) == 0:
        return None

    target_names = (
        [str(x) for x in label_target_names]
        if isinstance(label_target_names, list)
        else [f'prop_{i}' for i in label_target_indices]
    )

    batch_size = int(config.get('batch_size', 64))
    train_df = _predict_label_outputs_for_split(
        model=model,
        x_tokens=train_molecules_input,
        cond_norm=train_labels,
        lengths=train_length,
        seq_output_tokens=train_molecules_output,
        labels_raw=train_labels_raw,
        split_name='train',
        charset=np.asarray(charset),
        label_target_indices=list(label_target_indices),
        label_target_names=target_names,
        label_target_scale_raw=bool(label_targets_use_raw_scale),
        prop_norm_mean=np.asarray(prop_mean, dtype=np.float32),
        prop_norm_std=np.asarray(prop_std, dtype=np.float32),
        batch_size=batch_size,
    )
    validation_df = _predict_label_outputs_for_split(
        model=model,
        x_tokens=test_molecules_input,
        cond_norm=test_labels,
        lengths=test_length,
        seq_output_tokens=test_molecules_output,
        labels_raw=test_labels_raw,
        split_name='validation',
        charset=np.asarray(charset),
        label_target_indices=list(label_target_indices),
        label_target_names=target_names,
        label_target_scale_raw=bool(label_targets_use_raw_scale),
        prop_norm_mean=np.asarray(prop_mean, dtype=np.float32),
        prop_norm_std=np.asarray(prop_std, dtype=np.float32),
        batch_size=batch_size,
    )

    eval_df = pd.concat([train_df, validation_df], axis=0, ignore_index=True, sort=False)
    out_name = str(config.get('prediction_eval_filename', 'train_validation_prediction_eval.csv')).strip()
    if out_name == '':
        out_name = 'train_validation_prediction_eval.csv'
    out_path = os.path.abspath(os.path.join(config['save_dir'], out_name))
    eval_df.to_csv(out_path, index=False)
    print(f'saved train/validation prediction eval CSV: {out_path} (rows={len(eval_df)})')
    return out_path

# NOTE: compose_train_config_from_dict() may flatten/override nested config values.
# Capture optional label-prediction settings here so they remain stable.
predict_labels_cfg = bool(config.get('model', {}).get('predict_labels', False))
label_target_indices_cfg = config.get('model', {}).get('label_target_indices', None)
label_dim_cfg = config.get('model', {}).get('label_dim', None)
label_loss_weight_cfg = float(
    config.get('model', {}).get(
        'label_loss_weight',
        config.get('model', {}).get('lambda_label', 1.0),
    )
)
include_condition_in_label_head_cfg = bool(config.get('model', {}).get('include_condition_in_label_head', False))
label_targets_use_raw_scale_cfg = bool(config.get('model', {}).get('label_targets_use_raw_scale', False))
external_test_prop_file_cfg = config.get('data', {}).get('test_prop_file', None)

config = compose_train_config_from_dict(config)
config = apply_training_preset(config)
config = compose_train_config_from_dict(config)

base_save_dir = config['save_dir']
config['save_dir'] = build_train_run_save_dir(
    base_save_dir,
    run_name=config.get('run_name'),
    use_run_subdir=bool(config.get('use_run_subdir', True)),
)
print(f"save root dir: {base_save_dir}")
print(f"run save dir: {config['save_dir']}")

# check for gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# check so attention heads divide unit size evenly for transformer model
if config['model_mode'] == 'transformer':
    if config['unit_size'] % config['transformer_heads'] != 0:
        print(f"Possible values: unit_size={config['unit_size']}, transformer_heads={config['transformer_heads']}")
        raise ValueError(f'For transformer model, unit_size ({config["unit_size"]}) must be divisible by transformer_heads ({config["transformer_heads"]}).')


print (config)
#convert smiles to numpy array
# we will have two version of output, one with start and end token, one without. the one with start and end token is used for training, the one without is used for testing.
molecules_input, molecules_output, char, vocab, labels, length = load_data(config['prop_file'], config['seq_length'])
vocab_size = len(char)
if labels.ndim != 2 or labels.shape[1] == 0:
    raise ValueError('Property file must contain at least one numeric conditioning column after SMILES.')
config['num_prop'] = int(labels.shape[1])
print(f'inferred num_prop from {config["prop_file"]}: {config["num_prop"]}')

# Optional label prediction target configuration.
# By default, we reuse the same normalized property vector both as:
#   - conditioning input: c
#   - prediction target:  y_label
# If you want separate targets, split `labels` into two parts here.
predict_labels = bool(predict_labels_cfg)

num_prop = int(config['num_prop'])
default_prop_names = load_condition_property_names(config['prop_file'], num_prop)

label_target_indices = None
label_target_names = None
label_dim = 0
if predict_labels:
    if label_target_indices_cfg is None:
        # Default behavior for this project: if properties are [MW, LogP],
        # predict LogP only by default.
        label_target_indices = ([1] if num_prop == 2 else list(range(num_prop)))
    else:
        if not isinstance(label_target_indices_cfg, (list, tuple)) or len(label_target_indices_cfg) == 0:
            raise ValueError('label_target_indices must be a non-empty list of ints (or None).')
        label_target_indices = [int(i) for i in label_target_indices_cfg]
        for i in label_target_indices:
            if i < 0 or i >= num_prop:
                raise ValueError(
                    f'label_target_indices contains out-of-range index {i}. Must be in [0, {num_prop - 1}].'
                )

    label_target_names = [default_prop_names[i] if i < len(default_prop_names) else f'prop_{i}' for i in label_target_indices]

    if label_dim_cfg is None:
        label_dim = int(len(label_target_indices))
    else:
        label_dim = int(label_dim_cfg)
        if label_dim != int(len(label_target_indices)):
            raise ValueError(
                f'label_dim ({label_dim}) must match len(label_target_indices) ({len(label_target_indices)}).'
            )

    if label_dim <= 0:
        raise ValueError('label_dim must be > 0 when predict_labels=True')
    print(f'label prediction enabled: targets={label_target_names} (indices={label_target_indices})')

# divide data into training and test set
# can leak...
# could look into scaffold splitting for this!
if external_test_prop_file_cfg:
    print('using external test split file to prevent train/test leakage')
    print(f'  train file: {config["prop_file"]}')
    print(f'  test file : {external_test_prop_file_cfg}')
    print(
        f'  note: train_ratio={config.get("train_ratio")} is ignored because external test split is active.'
    )
    train_molecules_input = molecules_input
    train_molecules_output = molecules_output
    train_labels = labels
    train_length = length

    test_molecules_input, test_molecules_output, test_labels, test_length = _load_data_with_existing_vocab(
        str(external_test_prop_file_cfg),
        int(config['seq_length']),
        vocab,
    )
    if int(test_labels.shape[1]) != int(config['num_prop']):
        raise ValueError(
            f'external test split has {int(test_labels.shape[1])} properties, '
            f'but train split has {int(config["num_prop"])}.'
        )
else:
    train_molecules_input, test_molecules_input = split_train_test(molecules_input, config['train_ratio'])
    train_molecules_output, test_molecules_output = split_train_test(molecules_output, config['train_ratio'])
    train_labels, test_labels = split_train_test(labels, config['train_ratio'])
    train_length, test_length = split_train_test(length, config['train_ratio'])
    print(f'random split active via train_ratio={config["train_ratio"]}.')

num_train_before_aug = int(len(train_molecules_input))
smiles_aug_duplicates = int(config.get('smiles_augmentation_duplicates', 0))
if smiles_aug_duplicates > 0:
    train_molecules_input, train_molecules_output, train_labels, train_length, aug_stats = _augment_train_split_with_random_smiles(
        train_input=train_molecules_input,
        train_output=train_molecules_output,
        train_labels=train_labels,
        train_length=train_length,
        charset=char,
        vocab=vocab,
        seq_length=int(config['seq_length']),
        duplicates_per_smiles=smiles_aug_duplicates,
    )
    print(
        'smiles augmentation: '
        f'duplicates_per_smiles={smiles_aug_duplicates}, '
        f'train_before={num_train_before_aug}, train_after={len(train_molecules_input)}, '
        f'added={aug_stats["aug_added"]}, '
        f'skipped_invalid={aug_stats["skipped_invalid"]}, '
        f'skipped_too_long={aug_stats["skipped_too_long"]}, '
        f'skipped_unknown_token={aug_stats["skipped_unknown_token"]}'
    )
else:
    print('smiles augmentation disabled (smiles_augmentation_duplicates=0).')

# Keep unnormalized properties around if we want the label head to predict in
# original units (e.g. LogP). Conditioning still uses the normalized vector for
# numerical stability (especially with Transformer + AMP).
train_labels_raw = train_labels
test_labels_raw = test_labels

# Normalize conditioning properties (MW/LogP/etc). Unnormalized properties can be large and
# are a common cause of fp16 overflow -> NaN loss (especially with Transformer + AMP).
# this was also done in paper...
prop_mean = np.mean(train_labels, axis=0)
prop_std = np.std(train_labels, axis=0)
prop_std = np.where(prop_std < 1e-8, 1.0, prop_std)
train_labels = (train_labels - prop_mean) / prop_std
test_labels = (test_labels - prop_mean) / prop_std
print(f'property normalization: mean={prop_mean.tolist()} std={prop_std.tolist()}')

model_config = get_model_config(config, vocab_size=vocab_size)
model_config['smiles_augmentation_duplicates'] = int(smiles_aug_duplicates)
model_config['num_train_before_augmentation'] = int(num_train_before_aug)
model_config['num_train_after_augmentation'] = int(len(train_molecules_input))
model_config['prop_norm_mean'] = prop_mean.astype(np.float32).tolist()
model_config['prop_norm_std'] = prop_std.astype(np.float32).tolist()
model_config['condition_property_names'] = list(default_prop_names)
model_config['external_test_prop_file'] = str(external_test_prop_file_cfg) if external_test_prop_file_cfg else None
model_config['split_strategy'] = 'external_test_file' if external_test_prop_file_cfg else 'random_train_ratio'

# Wire optional label predictor settings into the model config.
model_config['predict_labels'] = bool(predict_labels)
model_config['label_dim'] = int(label_dim) if predict_labels else 0
model_config['label_loss_weight'] = float(label_loss_weight_cfg)
model_config['include_condition_in_label_head'] = bool(include_condition_in_label_head_cfg)
model_config['label_target_scale'] = 'raw' if bool(label_targets_use_raw_scale_cfg) else 'normalized'
if predict_labels:
    # These are guaranteed to be populated when predict_labels=True.
    assert label_target_indices is not None
    assert label_target_names is not None
    model_config['label_target_indices'] = list(label_target_indices)
    model_config['label_target_names'] = list(label_target_names)

#make save_dir
ensure_dir(config['save_dir'])

# save a single source of truth for recreating the trained model (including prop normalization)
training_config_path = save_training_config(model_config, config['save_dir'])
print(f'saved training config to: {training_config_path}')

model = CVAE(vocab_size, model_config)
print('Number of parameters : ', sum(p.numel() for p in model.parameters() if p.requires_grad))

scheduler = None
if config['use_reduce_lr_on_plateau']:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        model.optimizer,
        mode='min',
        factor=config['lr_plateau_factor'],
        patience=config['lr_plateau_patience'],
        threshold=config['lr_plateau_threshold'],
        min_lr=config['lr_plateau_min_lr'],
    )

history = {
    'train_loss': [],
    'test_loss': [],
    'lr': [],
}

if predict_labels:
    history['train_label_loss'] = []
    history['test_label_loss'] = []
    # MAE reported in *raw/original property units* (e.g. LogP units).
    # This stays interpretable regardless of whether the label head is trained on
    # normalized or raw targets.
    history['train_label_mae_raw'] = []
    history['test_label_mae_raw'] = []

best_test_loss = float('inf')
best_epoch = -1
epochs_without_improvement = 0
best_state_dict = None



# For logging time:
start_time = time.time()

# Main train loop. Will save checkpoints and history csv at the end of every epoch, and also on early stopping.

for epoch in range(config['num_epochs']):

    st = time.time()
    train_loss = []
    test_loss = []
    # reconstruction and KL losses logged to keep track of training dynam
    # KL loss != overall loss.

    train_recon = [] # reconstruction loss
    train_kl = [] # KL divergence loss
    train_mean_abs = [] # mean absolute value of latent vector z (to monitor KL collapse and latent space usage)
    train_log_sigma_mean = [] # mean of log sigma values from reparameterization (to monitor scale of latent space and potential fp16 overflow issues)
    train_log_sigma_min = [] # min of log sigma values from reparameterization (to monitor potential fp16 overflow issues)
    train_log_sigma_max = [] # max of log sigma values from reparameterization (to monitor potential fp16 overflow issues)
    train_grad_norm = [] # gradient norm for stability monitoring (especially important for transformer + amp)

    train_label_loss = [] # auxiliary label prediction loss, if enabled
    train_label_mae_raw = [] # auxiliary label MAE in raw/original units

    test_recon = [] # reconstruction loss
    test_kl = [] # KL divergence loss

    test_label_loss = [] # auxiliary label prediction loss on test set, if enabled
    test_label_mae_raw = [] # auxiliary label MAE in raw/original units

    beta = get_kl_beta(epoch, config)

    train_perm = np.random.permutation(len(train_molecules_input))


    # TRAIN LOOP:
    for start in range(0, len(train_perm), config['batch_size']):
        batch_idx = train_perm[start:start + config['batch_size']]
        x = train_molecules_input[batch_idx] # input with X start and E end token
        y = train_molecules_output[batch_idx] # output with only E end token (no start token, shifted by one position compared to input)
        l = train_length[batch_idx] # length of each sequence (without padding)
        c = train_labels[batch_idx] # conditioning properties (normalized)

        # Label prediction targets:
        # - Default: normalized targets (same scale as conditioning `c`).
        # - Optional: raw targets for interpretability in original units.
        y_label = None
        if predict_labels:
            assert label_target_indices is not None
            if bool(label_targets_use_raw_scale_cfg):
                y_label = train_labels_raw[batch_idx][:, label_target_indices]
            else:
                y_label = c[:, label_target_indices]
        metrics = model.train_batch(x, y, l, c, y_label=y_label, beta=beta, return_metrics=True)
        # check that metrics is a dict and contains expected keys
        if not isinstance(metrics, dict):
            raise TypeError('train_batch(return_metrics=True) must return a metrics dict.')
        train_loss.append(metrics['total_loss'])
        train_recon.append(metrics['recon_loss'])
        train_kl.append(metrics['kl_loss'])
        if predict_labels:
            train_label_loss.append(metrics.get('label_loss', 0.0))
            # Convert per-dim MAE to raw units if the label targets were normalized.
            # For normalized targets: y_norm = (y_raw - mean) / std
            # => |y_raw - y_true_raw| = |y_norm - y_true_norm| * std
            mae_per_dim = metrics.get('label_mae_per_dim')
            if mae_per_dim is None:
                # Backward-compatible fallback: use scalar MAE.
                mae_scalar = float(metrics.get('label_mae', 0.0))
                if bool(label_targets_use_raw_scale_cfg):
                    train_label_mae_raw.append(mae_scalar)
                else:
                    std_arr = np.asarray(prop_std, dtype=np.float32)[label_target_indices]
                    train_label_mae_raw.append(float(mae_scalar * float(np.mean(std_arr))))
            else:
                mae_vec = np.asarray(mae_per_dim, dtype=np.float32)
                if bool(label_targets_use_raw_scale_cfg):
                    train_label_mae_raw.append(float(np.mean(mae_vec)))
                else:
                    std_arr = np.asarray(prop_std, dtype=np.float32)[label_target_indices]
                    std_arr = np.where(std_arr < 1e-8, 1.0, std_arr)
                    train_label_mae_raw.append(float(np.mean(mae_vec * std_arr)))
        train_mean_abs.append(metrics['mean_abs'])
        train_log_sigma_mean.append(metrics['log_sigma_mean'])
        train_log_sigma_min.append(metrics['log_sigma_min'])
        train_log_sigma_max.append(metrics['log_sigma_max'])
        train_grad_norm.append(metrics['grad_norm'])

    #    
    # test on test set (trend monitoring).
    test_perm = np.random.permutation(len(test_molecules_input))
    for start in range(0, len(test_perm), config['batch_size']):
        batch_idx = test_perm[start:start + config['batch_size']]
        x = test_molecules_input[batch_idx] # input with X start and E end token
        y = test_molecules_output[batch_idx] # output with only E end token (no start token, shifted by one position compared to input)
        l = test_length[batch_idx] # length of each sequence (without padding)
        c = test_labels[batch_idx] # conditioning properties (normalized)

        y_label = None
        if predict_labels:
            assert label_target_indices is not None
            if bool(label_targets_use_raw_scale_cfg):
                y_label = test_labels_raw[batch_idx][:, label_target_indices]
            else:
                y_label = c[:, label_target_indices]
        metrics = model.test_batch(x, y, l, c, y_label=y_label, beta=beta, return_metrics=True)
        if not isinstance(metrics, dict):
            raise TypeError('test_batch(return_metrics=True) must return a metrics dict.')
        test_loss.append(metrics['total_loss'])
        test_recon.append(metrics['recon_loss'])
        test_kl.append(metrics['kl_loss'])
        if predict_labels:
            test_label_loss.append(metrics.get('label_loss', 0.0))
            mae_per_dim = metrics.get('label_mae_per_dim')
            if mae_per_dim is None:
                mae_scalar = float(metrics.get('label_mae', 0.0))
                if bool(label_targets_use_raw_scale_cfg):
                    test_label_mae_raw.append(mae_scalar)
                else:
                    std_arr = np.asarray(prop_std, dtype=np.float32)[label_target_indices]
                    test_label_mae_raw.append(float(mae_scalar * float(np.mean(std_arr))))
            else:
                mae_vec = np.asarray(mae_per_dim, dtype=np.float32)
                if bool(label_targets_use_raw_scale_cfg):
                    test_label_mae_raw.append(float(np.mean(mae_vec)))
                else:
                    std_arr = np.asarray(prop_std, dtype=np.float32)[label_target_indices]
                    std_arr = np.where(std_arr < 1e-8, 1.0, std_arr)
                    test_label_mae_raw.append(float(np.mean(mae_vec * std_arr)))
    

    train_loss = np.mean(np.array(train_loss))
    test_loss = np.mean(np.array(test_loss))
    
    #log_cuda_mem(prefix=f"[epoch {epoch}]")

    #stability check, stop train if non-finite loss..
    if not np.isfinite(train_loss) or not np.isfinite(test_loss):
        print(f'non-finite loss detected at epoch {epoch} (train={train_loss}, test={test_loss}), stopping early')
        break

    if scheduler is not None:
        scheduler.step(float(test_loss))

    current_lr = model.optimizer.param_groups[0]['lr']

    history['train_loss'].append(train_loss)
    history['test_loss'].append(test_loss)
    history['lr'].append(current_lr)

    if predict_labels:
        history['train_label_loss'].append(float(np.mean(np.array(train_label_loss))) if len(train_label_loss) else 0.0)
        history['test_label_loss'].append(float(np.mean(np.array(test_label_loss))) if len(test_label_loss) else 0.0)
        history['train_label_mae_raw'].append(float(np.mean(np.array(train_label_mae_raw))) if len(train_label_mae_raw) else 0.0)
        history['test_label_mae_raw'].append(float(np.mean(np.array(test_label_mae_raw))) if len(test_label_mae_raw) else 0.0)

    # robust early stopping with:
    # min_delta: least improvment to count as an improvment.
    # patience: number of epochs to wait for improvment before stopping.
    # restore_best: whether to restore model weights from the epoch with the best test loss at the end of training.
    improved = float(test_loss) < (best_test_loss - float(config['early_stopping_min_delta']))
    if improved:
        best_test_loss = float(test_loss)
        best_epoch = epoch
        epochs_without_improvement = 0
        if config['early_stopping_restore_best']:
            best_state_dict = deepcopy(model.state_dict())
            save_best_checkpoint(
                epoch=epoch,
                config=config,
                model=model,
                model_config=model_config,
                best_state_dict=best_state_dict,
                best_epoch=best_epoch,
            )
    else:
        epochs_without_improvement += 1
        print(f"epochs without improvement: {epochs_without_improvement}/{config['early_stopping_patience']} (best epoch: {best_epoch}, best test loss: {best_test_loss:.6f})")


    #NOTE: Early stopping will trigger if no improvement  
    if epochs_without_improvement >= config['early_stopping_patience']:
        print(
            f'early stop at epoch {epoch} since no improvement for '
            f'{epochs_without_improvement} epochs (best epoch: {best_epoch}, best test loss: {best_test_loss:.6f})'
        )

        # Save the current weights at early-stop epoch for traceability.
        if bool(config.get('save_early_stop_checkpoint', True)):
            save_current_checkpoint(
                epoch=epoch,
                config=config,
                model=model,
                model_config=model_config,
                suffix='_early_stop',
            )
        if bool(config.get('save_history_csv', True)):
            save_history_csv(config=config, history=history)
        break
    # end time for epoch
    end = time.time()  
    passed_time = end - st

    time_per_epoch = (end - start_time) / (epoch + 1)
    expected_time_remaining = time_per_epoch * (config['num_epochs'] - epoch - 1)

      
    if epoch==0:
        print(f"{'Epoch':<10}{'Train Loss':<15}{'Test Loss':<15}{'Learning Rate':<15}{'Time (s)':<10}{'ETA (min)':<10}")
    print(f"{epoch:<10}{train_loss:<15.3f}{test_loss:<15.3f}{current_lr:<15.6f}{passed_time:<10.3f}{expected_time_remaining/60:<10.2f}")
    if epoch % int(config.get('diagnostics_every', 1)) == 0:
        label_diag = ""
        if predict_labels:
            label_diag = (
                f" label_loss(train/test)="
                f"{np.mean(train_label_loss) if len(train_label_loss) else 0.0:.4f}/"
                f"{np.mean(test_label_loss) if len(test_label_loss) else 0.0:.4f}"
                f" label_mae_raw(train/test)="
                f"{float(np.mean(train_label_mae_raw)) if len(train_label_mae_raw) else 0.0:.4f}/"
                f"{float(np.mean(test_label_mae_raw)) if len(test_label_mae_raw) else 0.0:.4f}"
            )
        print(
            f"diag epoch={epoch} beta={beta:.3f} "
            f"train_recon={np.mean(train_recon):.4f} train_kl={np.mean(train_kl):.4f} "
            f"test_recon={np.mean(test_recon):.4f} test_kl={np.mean(test_kl):.4f} "
            f"{label_diag} "
            f"mean_abs={np.mean(train_mean_abs):.4f} "
            f"log_sigma(mean/min/max)={np.mean(train_log_sigma_mean):.4f}/"
            f"{np.mean(train_log_sigma_min):.4f}/{np.mean(train_log_sigma_max):.4f} "
            f"grad_norm={np.mean(train_grad_norm):.4f}"
        )

    is_last_epoch = epoch == (config['num_epochs'] - 1)
    save_epoch = (epoch + 1) % config['save_every'] == 0

    # Occaisonal_checkpointing
    if is_last_epoch:
        # Always save current weights for the last epoch if training reaches it.
        if bool(config.get('save_final_checkpoint', True)):
            save_current_checkpoint(
                epoch=epoch,
                config=config,
                model=model,
                model_config=model_config,
                suffix='_final',
            )
        if bool(config.get('save_history_csv', True)):
            save_history_csv(config=config, history=history)
    elif save_epoch and bool(config.get('save_periodic_checkpoints', True)):
        # Occasional save of current epoch weights.
        save_current_checkpoint(
            epoch=epoch,
            config=config,
            model=model,
            model_config=model_config,
            suffix='_periodic',
        )
        if bool(config.get('save_history_csv', True)):
            save_history_csv(config=config, history=history)

# Export split-level prediction-vs-ground-truth CSV for downstream residual/error plots.
if predict_labels:
    if bool(config.get('early_stopping_restore_best', True)) and best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    _export_train_validation_prediction_eval_csv(
        model=model,
        config=config,
        charset=np.asarray(char),
        train_molecules_input=train_molecules_input,
        train_molecules_output=train_molecules_output,
        train_length=train_length,
        train_labels=train_labels,
        train_labels_raw=train_labels_raw,
        test_molecules_input=test_molecules_input,
        test_molecules_output=test_molecules_output,
        test_length=test_length,
        test_labels=test_labels,
        test_labels_raw=test_labels_raw,
        label_target_indices=label_target_indices,
        label_target_names=label_target_names,
        label_targets_use_raw_scale=bool(label_targets_use_raw_scale_cfg),
        prop_mean=np.asarray(prop_mean, dtype=np.float32),
        prop_std=np.asarray(prop_std, dtype=np.float32),
    )

