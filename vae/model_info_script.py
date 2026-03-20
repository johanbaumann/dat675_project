#!/usr/bin/env python
"""
Model information and parameter counting script.

This script:
1. Loads configuration from fold_pipeline_config.example.json
2. Initializes the CVAE model
3. Prints detailed parameter breakdown per layer and overall model info
"""

import json
import sys
import os
import argparse
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models.model_labels import CVAE


_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_DEFAULT_FOLD_CONFIG = os.path.join(_THIS_DIR, 'fold_pipeline', 'fold_pipeline_config.example.json')


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Print CVAE model parameter breakdown from fold pipeline config.')
    parser.add_argument('--config', type=str, default=_DEFAULT_FOLD_CONFIG)
    return parser.parse_args()


def _resolve_required_path(path_value: str, *, base_dir: str, key_name: str) -> str:
    raw = str(path_value).strip()
    if raw == '':
        raise ValueError(f'{key_name} cannot be empty')
    if os.path.isabs(raw):
        return os.path.abspath(raw)
    return os.path.abspath(os.path.join(base_dir, raw))


def load_config(config_path: str) -> dict:
    """Load configuration from fold_pipeline_config.example.json."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def build_model_config(fold_config: dict) -> dict:
    """Extract and build model config from fold_pipeline config."""
    train_config = fold_config.get('train', {})
    base_config = train_config.get('base_config', {})
    
    model_cfg = base_config.get('model', {})
    transformer_cfg = base_config.get('transformer', {})
    
    # Helper to get value or default if None
    def _get_or_default(val, default):
        return default if val is None else val
    
    # Build the args dict that CVAE expects
    num_prop = _get_or_default(model_cfg.get('num_prop'), 2)
    label_dim = _get_or_default(model_cfg.get('label_dim'), 0)
    label_target_indices = _get_or_default(model_cfg.get('label_target_indices'), None)
    predict_labels = model_cfg.get('predict_labels', False)
    
    # If predict_labels is True but label_dim is not set, default to num_prop (or 1 if num_prop is None)
    if predict_labels and (label_dim is None or label_dim == 0):
        label_dim = num_prop if num_prop else 1
    
    args = {
        'vocab_size': 36,  # SMILES vocabulary size (fixed)
        'batch_size': base_config.get('training', {}).get('batch_size', 64),
        'latent_size': model_cfg.get('latent_size', 200),
        'unit_size': model_cfg.get('unit_size', 256),
        'n_rnn_layer': model_cfg.get('n_rnn_layer', 3),
        'seq_length': base_config.get('data', {}).get('seq_length', 120),
        'mean': model_cfg.get('mean', 0.0),
        'stddev': model_cfg.get('stddev', 1.0),
        'lr': base_config.get('optimization', {}).get('lr', 1e-4),
        'num_prop': num_prop,
        'model_mode': model_cfg.get('mode', 'lstm').lower(),
        'optimizer': base_config.get('optimization', {}).get('optimizer', 'adam'),
        'weight_decay': base_config.get('optimization', {}).get('weight_decay', 0.0),
        'use_amp': base_config.get('optimization', {}).get('use_amp', False),
        'amp_dtype': base_config.get('optimization', {}).get('amp_dtype', 'float16'),
        'grad_clip_norm': base_config.get('optimization', {}).get('grad_clip_norm', 1.0),
        'transformer_heads': transformer_cfg.get('heads', 8),
        'transformer_ff_size': transformer_cfg.get('ff_size', 1024),
        'transformer_dropout': transformer_cfg.get('dropout', 0.1),
        'predict_labels': predict_labels,
        'label_dim': int(label_dim) if label_dim else 0,
        'label_target_indices': label_target_indices,
        'label_loss_weight': model_cfg.get('label_loss_weight', 1.0),
        'include_condition_in_label_head': model_cfg.get('include_condition_in_label_head', False),
    }
    
    return args


def main():
    args = _parse_args()
    config_path = _resolve_required_path(args.config, base_dir=os.getcwd(), key_name='--config')
    
    if not os.path.isfile(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    print("=" * 80)
    print("CVAE Model Information Script")
    print("=" * 80)
    
    # Load config
    fold_config = load_config(config_path)
    args = build_model_config(fold_config)
    
    print(f"\nConfiguration loaded from: {config_path}")
    print("\nModel Configuration:")
    print(f"  model_mode:              {args['model_mode']}")
    print(f"  vocab_size:              {args['vocab_size']}")
    print(f"  latent_size:             {args['latent_size']}")
    print(f"  unit_size:               {args['unit_size']}")
    print(f"  n_rnn_layer:             {args['n_rnn_layer']}")
    print(f"  seq_length:              {args['seq_length']}")
    print(f"  num_prop:                {args['num_prop']}")
    print(f"  optimizer:               {args['optimizer']}")
    print(f"  learning_rate:           {args['lr']}")
    print(f"  weight_decay:            {args['weight_decay']}")
    print(f"  grad_clip_norm:          {args['grad_clip_norm']}")
    print(f"  use_amp:                 {args['use_amp']}")
    
    if args['model_mode'] == 'transformer':
        print(f"  transformer_heads:       {args['transformer_heads']}")
        print(f"  transformer_ff_size:     {args['transformer_ff_size']}")
        print(f"  transformer_dropout:     {args['transformer_dropout']}")
    
    if args['predict_labels']:
        print(f"  predict_labels:          True")
        print(f"  label_dim:               {args['label_dim']}")
        print(f"  label_loss_weight:       {args['label_loss_weight']}")
        print(f"  include_condition:       {args['include_condition_in_label_head']}")
    
    # Initialize model
    print("\nInitializing model...")
    model = CVAE(vocab_size=args['vocab_size'], args=args)
    
    # Print detailed parameter info
    print("\n" + "=" * 80)
    print("DETAILED PARAMETER BREAKDOWN")
    print("=" * 80)
    
    model.print_parameters_info()


if __name__ == '__main__':
    main()
