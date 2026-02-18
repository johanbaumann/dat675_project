from model import CVAE
from utils import *
import numpy as np
import time
import pandas as pd
import torch
from copy import deepcopy


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
            'weight_decay': 0.0,
            'use_amp': True,
            'kl_anneal_enabled': True,
            'kl_anneal_start_beta': 0.01,
            'kl_anneal_max_beta': 1.0,
            'kl_anneal_hold_epochs': 0,
            'kl_anneal_warmup_epochs': 50,
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
        'prop_file': 'prop_mw_logp.txt',
        'seq_length': 120,
        'train_ratio': 0.75,
    },
    'model': {
        'mode': 'lstm',  # 'lstm' or 'transformer'
        'latent_size': 200,
        'unit_size': 512,
        'n_rnn_layer': 3,
        'mean': 0.0,
        'stddev': 1.0,
        'num_prop': None,  # inferred from property file
    },
    'transformer': {
        'heads': 8,
        'ff_size': 1024,
        'dropout': 0.15,
    },
    'optimization': {
        'optimizer': 'adam',
        'lr': 0.0001, # 10e-4
        'weight_decay': 0.0,
        'use_amp': False,
        'amp_dtype': 'float16',
        'grad_clip_norm': 4.0,
    },
    'training': {
        'batch_size': 128,
        'num_epochs': 100,
        'save_dir': 'save/',
        'save_every': 10,
        'early_stopping_patience': 10,
        'early_stopping_min_delta': 0.0,
        'early_stopping_restore_best': True,
    },
    'scheduler': {
        'enabled': True,
        'factor': 0.5,
        'patience': 5,
        'threshold': 1e-4,
        'min_lr': 1e-6,
    },
    'kl': {
        'enabled': False,
        'start_beta': 0.0,
        'max_beta': 1.0,
        'hold_epochs': 0,
        'warmup_epochs': 50,
    },
    'diagnostics': {
        'every': 1,
    },
}

config = compose_train_config_from_dict(config)
config = apply_training_preset(config)
config = compose_train_config_from_dict(config)

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

#divide data into training and test set
# can leak...
# could look into scaffold splitting for this!
train_molecules_input, test_molecules_input = split_train_test(molecules_input, config['train_ratio'])
train_molecules_output, test_molecules_output = split_train_test(molecules_output, config['train_ratio'])
train_labels, test_labels = split_train_test(labels, config['train_ratio'])
train_length, test_length = split_train_test(length, config['train_ratio'])

# Normalize conditioning properties (MW/LogP/etc). Unnormalized properties can be large and
# are a common cause of fp16 overflow -> NaN loss (especially with Transformer + AMP).
prop_mean = np.mean(train_labels, axis=0)
prop_std = np.std(train_labels, axis=0)
prop_std = np.where(prop_std < 1e-8, 1.0, prop_std)
train_labels = (train_labels - prop_mean) / prop_std
test_labels = (test_labels - prop_mean) / prop_std
print(f'property normalization: mean={prop_mean.tolist()} std={prop_std.tolist()}')

model_config = get_model_config(config, vocab_size=vocab_size)
model_config['prop_norm_mean'] = prop_mean.astype(np.float32).tolist()
model_config['prop_norm_std'] = prop_std.astype(np.float32).tolist()

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

best_test_loss = float('inf')
best_epoch = -1
epochs_without_improvement = 0
best_state_dict = None

start_time = time.time()

for epoch in range(config['num_epochs']):

    st = time.time()
    train_loss = []
    test_loss = []
    train_recon = []
    train_kl = []
    train_mean_abs = []
    train_log_sigma_mean = []
    train_log_sigma_min = []
    train_log_sigma_max = []
    train_grad_norm = []

    test_recon = []
    test_kl = []

    beta = get_kl_beta(epoch, config)

    train_perm = np.random.permutation(len(train_molecules_input))


    # TRAIN LOOP:
    for start in range(0, len(train_perm), config['batch_size']):
        batch_idx = train_perm[start:start + config['batch_size']]
        x = train_molecules_input[batch_idx]
        y = train_molecules_output[batch_idx]
        l = train_length[batch_idx]
        c = train_labels[batch_idx]
        metrics = model.train_batch(x, y, l, c, beta=beta, return_metrics=True)
        if not isinstance(metrics, dict):
            raise TypeError('train_batch(return_metrics=True) must return a metrics dict.')
        train_loss.append(metrics['total_loss'])
        train_recon.append(metrics['recon_loss'])
        train_kl.append(metrics['kl_loss'])
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
        x = test_molecules_input[batch_idx]
        y = test_molecules_output[batch_idx]
        l = test_length[batch_idx]
        c = test_labels[batch_idx]
        metrics = model.test_batch(x, y, l, c, beta=beta, return_metrics=True)
        if not isinstance(metrics, dict):
            raise TypeError('test_batch(return_metrics=True) must return a metrics dict.')
        test_loss.append(metrics['total_loss'])
        test_recon.append(metrics['recon_loss'])
        test_kl.append(metrics['kl_loss'])
    

    train_loss = np.mean(np.array(train_loss))
    test_loss = np.mean(np.array(test_loss))
    
    log_cuda_mem(prefix=f"[epoch {epoch}]")

    if not np.isfinite(train_loss) or not np.isfinite(test_loss):
        print(f'non-finite loss detected at epoch {epoch} (train={train_loss}, test={test_loss}), stopping early')
        break

    if scheduler is not None:
        scheduler.step(float(test_loss))

    current_lr = model.optimizer.param_groups[0]['lr']

    history['train_loss'].append(train_loss)
    history['test_loss'].append(test_loss)
    history['lr'].append(current_lr)

    # robust early stopping with min_delta
    improved = float(test_loss) < (best_test_loss - float(config['early_stopping_min_delta']))
    if improved:
        best_test_loss = float(test_loss)
        best_epoch = epoch
        epochs_without_improvement = 0
        if config['early_stopping_restore_best']:
            best_state_dict = deepcopy(model.state_dict())
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= config['early_stopping_patience']:
        print(
            f'early stop at epoch {epoch} since no improvement for '
            f'{epochs_without_improvement} epochs (best epoch: {best_epoch}, best test loss: {best_test_loss:.6f})'
        )

        if config['early_stopping_restore_best'] and best_state_dict is not None:
            model.load_state_dict(best_state_dict)
            print(f'restored best model weights from epoch {best_epoch}')
        
        ckpt_path = config['save_dir']+'/model_'+str(epoch)+'.ckpt'
        model.save(ckpt_path, epoch, model_config=model_config)
        history_df = pd.DataFrame(history)
        history_df.to_csv(config['save_dir']+'/history.csv', index=False)
        break
            
    end = time.time()  
    passed_time = end - st

    time_per_epoch = (end - start_time) / (epoch + 1)
    expected_time_remaining = time_per_epoch * (config['num_epochs'] - epoch - 1)

      
    if epoch==0:
        print(f"{'Epoch':<10}{'Train Loss':<15}{'Test Loss':<15}{'Learning Rate':<15}{'Time (s)':<10}{'ETA (min)':<10}")
    print(f"{epoch:<10}{train_loss:<15.3f}{test_loss:<15.3f}{current_lr:<15.6f}{passed_time:<10.3f}{expected_time_remaining/60:<10.2f}")
    if epoch % int(config.get('diagnostics_every', 1)) == 0:
        print(
            f"diag epoch={epoch} beta={beta:.3f} "
            f"train_recon={np.mean(train_recon):.4f} train_kl={np.mean(train_kl):.4f} "
            f"test_recon={np.mean(test_recon):.4f} test_kl={np.mean(test_kl):.4f} "
            f"mean_abs={np.mean(train_mean_abs):.4f} "
            f"log_sigma(mean/min/max)={np.mean(train_log_sigma_mean):.4f}/"
            f"{np.mean(train_log_sigma_min):.4f}/{np.mean(train_log_sigma_max):.4f} "
            f"grad_norm={np.mean(train_grad_norm):.4f}"
        )

    if epoch == config['num_epochs']-1 or (epoch + 1) % config['save_every'] == 0:
        if config['early_stopping_restore_best'] and best_state_dict is not None:
            model.load_state_dict(best_state_dict)
            print(f'training ended; restored best model weights from epoch {best_epoch}')
        ckpt_path = config['save_dir']+'/model_'+str(epoch)+'.ckpt'
        model.save(ckpt_path, epoch, model_config=model_config)
        #save history as csv file
        history_df = pd.DataFrame(history)
        history_df.to_csv(config['save_dir']+'/history.csv', index=False)

