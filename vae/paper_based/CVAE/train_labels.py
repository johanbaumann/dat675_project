"""train_label.py

Train script for CVAE with property conditioning.

This script now optionally trains an auxiliary label predictor head (see
`model_labels.py`) that predicts molecular labels from the latent vector `z`.

Important invariants (kept):
- `load_data()` signature and output semantics are unchanged.
- Conditioning vector `c` is unchanged (still taken from the property file).
- Sampling pipeline is unchanged.

CHANGELOG
---------
2026-02-23
- Added optional label predictor training via `predict_labels` config knobs.
- Uses `model_labels.CVAE` so the new head can be trained without changing the
    base `model.py` implementation.
"""

from model_labels import CVAE
from utils import *
import numpy as np
import time
import pandas as pd
import torch
from copy import deepcopy
import os
import glob


def log_cuda_mem(prefix: str = "") -> None:
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        print(f"{prefix} cuda_mem_allocated={alloc:.1f} MiB reserved={reserved:.1f} MiB")


def get_kl_beta(epoch:int, cfg:dict) -> float:
    if not bool(cfg.get('kl_anneal_enabled', False)):
        return 1.0
    start_beta = float(cfg.get('kl_anneal_start_beta', 0.0))
    max_beta = float(cfg.get('kl_anneal_max_beta', 1.0))
    hold_epochs = int(cfg.get('kl_anneal_hold_epochs', 0))
    warmup_epochs = max(1, int(cfg.get('kl_anneal_warmup_epochs', 20)))
    if epoch < hold_epochs:
        return start_beta
    progress = (epoch - hold_epochs) / float(warmup_epochs)
    progress = min(max(progress, 0.0), 1.0)
    beta = start_beta + (max_beta - start_beta) * progress
    return float(beta)


def apply_training_preset(cfg:dict) -> dict:
    preset = str(cfg.get('training_preset', 'custom')).strip().lower()
    if preset in ('', 'custom', 'none'):
        print('training preset: custom (no automatic overrides)')
        return cfg

    if preset == 'stable_transformer':
        cfg.update({
            'model_mode': 'transformer',
            'optimizer': 'adamw',
            'weight_decay': 0.001, # 
            'use_amp': True,
            'kl_anneal_enabled': True,
            'kl_anneal_start_beta': 0.01,
            'kl_anneal_max_beta': 1.0,
            'kl_anneal_hold_epochs': 0,
            'kl_anneal_warmup_epochs': 8,
            'diagnostics_every': 1,
        })
        print('training preset: stable_transformer (applied)')
        return cfg

    raise ValueError("training_preset must be one of: 'custom', 'stable_transformer'")

# Single source of truth for run configuration.
# Grouped sections are easier to edit; utils will flatten this to legacy keys.
config = {
    'training_preset': 'custom',  # 'custom' or 'stable_transformer'
    'data': {
        'prop_file': '250k_zinc_clean.txt',
        'seq_length': 120,
        'train_ratio': 0.75,
    },
    'model': {
        'mode': 'lstm',  # 'lstm' or 'transformer'
        'latent_size': 200,
        'unit_size': 512,
        'n_rnn_layer': 3, # 2 layers for transformers, 3 for lstm (memory constraints...)
        'mean': 0.0,
        'stddev': 1.0,
        'num_prop': None,  # inferred from property file

        # Optional multi-task head:
        # - predict_labels=False keeps the model identical to the base CVAE.
        # - When enabled, the model predicts a label vector from latent `z`.
        'predict_labels': True,
        # Which conditioning columns to predict as labels.
        # Example (for: MW, LogP setup): set to [1] to predict LogP only.
        # If None and predict_labels=True, defaults to predicting all properties.
        'label_target_indices': [1],  # predict LogP only by default for the MW/LogP setup
        'label_dim': None,  # if None, inferred from property file
        'label_loss_weight': 1.0, # relative weight of label prediction loss compared to reconstruction + KL loss (which are weighted by beta)

        # Optional label-head variants:
        # If True, the label head predicts from both (z, c) instead of z only.
        # This can be useful if you want predicted labels to track the sampling
        # target properties more directly.
        'include_condition_in_label_head': True,

        # If True, train the label head on *raw* (unnormalized) property values.
        # If False (default), train it on the normalized conditioning vector.
        'label_targets_use_raw_scale': False,
    },
    'transformer': {
        'heads': 8,
        'ff_size': 1024,
        'dropout': 0.15,
    },
    'optimization': {
        'optimizer': 'adam', # 'adam' for lstm, 'adamw' for transformer (with weight decay)
        'lr': 1e-3, # 10e-4, 1e-5 for transformer..
        'weight_decay': 0.000, # 0.001 for transformer 
        'use_amp': True, # true if using transformer with fp16, can cause instability with lstm
        'amp_dtype': 'bfloat16', #bfloat16 for transformer (since i have 3070)
        'grad_clip_norm': 4.0,
    },
    'training': {
        'batch_size': 128, # 64 for transformer... 128 for lstm
        'num_epochs': 100, # transformer need more
        'save_dir': 'save/',
        'run_name': None,  # If None, auto-generated timestamped run folder is used.
        'use_run_subdir': True,  # If True, save into save_dir/<run_name_or_timestamp>/
        'save_every': 10,
        'early_stopping_patience': 10,
        'early_stopping_min_delta': 0.001,
        'early_stopping_restore_best': True,
    },
    'scheduler': {
        'enabled': True,
        'factor': 0.5,
        'patience': 2,
        'threshold': 1e-3,
        'min_lr': 1e-6,
    },
    'kl': {
        'enabled': True,
        'start_beta': 1.0, # start with low KL weight to allow model to learn reconstruction before regularizing latent space, can help with stability (especially for transformer + amp).
        'max_beta': 2.0,
        'hold_epochs': 0,
        'warmup_epochs': 8,
    },
    'diagnostics': {
        'every': 1,
    },
}

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
default_prop_names = ['MW', 'LogP'] if num_prop == 2 else [f'prop_{i}' for i in range(num_prop)]

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

#divide data into training and test set
# can leak...
# could look into scaffold splitting for this!
train_molecules_input, test_molecules_input = split_train_test(molecules_input, config['train_ratio'])
train_molecules_output, test_molecules_output = split_train_test(molecules_output, config['train_ratio'])
train_labels, test_labels = split_train_test(labels, config['train_ratio'])
train_length, test_length = split_train_test(length, config['train_ratio'])

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
model_config['prop_norm_mean'] = prop_mean.astype(np.float32).tolist()
model_config['prop_norm_std'] = prop_std.astype(np.float32).tolist()

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


def save_history_csv(*, config: dict, history: dict) -> None:
    history_df = pd.DataFrame(history)
    history_df.to_csv(config['save_dir'] + '/history.csv', index=False)


def save_current_checkpoint(*, epoch: int, config: dict, model: CVAE, model_config: dict, suffix: str = "") -> None:
    """Save current in-memory model weights with epoch in filename."""
    ckpt_path = config['save_dir'] + f'/model_{epoch}{suffix}.ckpt'
    model.save(ckpt_path, epoch, model_config=model_config)


def _delete_previous_best_checkpoints(save_dir: str) -> None:
    """Keep only one rolling best checkpoint file on disk."""
    patterns = [
        os.path.join(save_dir, 'model_best.ckpt-*.pt'),
        os.path.join(save_dir, 'model_*best*.ckpt-*.pt'),
    ]
    deleted = set()
    for pattern in patterns:
        for path in glob.glob(pattern):
            if path in deleted:
                continue
            try:
                os.remove(path)
                deleted.add(path)
            except OSError:
                continue


def save_best_checkpoint(
    *,
    epoch: int,
    config: dict,
    model: CVAE,
    model_config: dict,
    best_state_dict,
    best_epoch: int,
) -> None:
    """Save a single rolling best checkpoint named model_best.ckpt-<best_epoch>.pt."""
    if best_state_dict is None:
        return

    _delete_previous_best_checkpoints(config['save_dir'])

    restore_after = deepcopy(model.state_dict())
    model.load_state_dict(best_state_dict)
    print(f'saving new best model from epoch {best_epoch} (found at epoch {epoch})')
    model.save(config['save_dir'] + '/model_best.ckpt', best_epoch, model_config=model_config)
    model.load_state_dict(restore_after)



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
        save_current_checkpoint(
            epoch=epoch,
            config=config,
            model=model,
            model_config=model_config,
            suffix='_early_stop',
        )
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
        save_current_checkpoint(
            epoch=epoch,
            config=config,
            model=model,
            model_config=model_config,
            suffix='_final',
        )
        save_history_csv(config=config, history=history)
    elif save_epoch:
        # Occasional save of current epoch weights.
        save_current_checkpoint(
            epoch=epoch,
            config=config,
            model=model,
            model_config=model_config,
            suffix='_periodic',
        )
        save_history_csv(config=config, history=history)

