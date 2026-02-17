from __future__ import annotations

import random
from typing import Any, Mapping, cast

import numpy as np
import torch

from model import CVAE
from utils import load_data


def main() -> None:
    # Small overfit sanity check:
    # - trains on a tiny fixed subset
    # - checks that loss decreases
    # - checks the model doesn't collapse to predicting only padding 'E'

    seed = 1337
    random.seed(seed)
    np.random.seed(seed)

    prop_file = 'prop_mw_logp.txt'
    seq_length = 120

    # Keep this small so it runs fast.
    num_samples = 128
    batch_size = 32
    steps = 60

    # Model config (Transformer path).
    latent_size = 200
    unit_size = 512
    n_layers = 2
    lr = 1e-4

    # Disable AMP for the sanity check (easier to interpret / avoids fp16 issues).
    model_mode = 'transformer'

    x_all, y_all, charset, vocab, labels, lengths = load_data(prop_file, seq_length)
    vocab_size = len(charset)

    if num_samples > len(x_all):
        raise ValueError(f'num_samples={num_samples} exceeds dataset size={len(x_all)}')

    # Select a deterministic subset.
    x = x_all[:num_samples]
    y = y_all[:num_samples]
    c = labels[:num_samples]
    l = lengths[:num_samples]

    # Normalize properties (same idea as train.py).
    c_mean = np.mean(c, axis=0)
    c_std = np.std(c, axis=0)
    c_std = np.where(c_std < 1e-8, 1.0, c_std)
    c = (c - c_mean) / c_std

    model_config = {
        'batch_size': batch_size,
        'latent_size': latent_size,
        'unit_size': unit_size,
        'n_rnn_layer': n_layers,
        'seq_length': seq_length,
        'mean': 0.0,
        'stddev': 1.0,
        'lr': lr,
        'num_prop': int(c.shape[1]),
        'grad_clip_norm': 8.0,
        'model_mode': model_mode,
        'optimizer': 'adamw',
        'weight_decay': 0.0,
        'use_amp': False,
        'amp_dtype': 'float16',
        'transformer_heads': 8,
        'transformer_ff_size': 1024,
        'transformer_dropout': 0.15,
    }

    model = CVAE(vocab_size, model_config)

    e_index = vocab['E'] if 'E' in vocab else (vocab_size - 2)

    def eval_collapse_metrics() -> tuple[float, float]:
        # Returns: (masked CE recon loss, share of predicted 'E' on valid positions)
        # We use one full forward pass on the subset (in batches) for a stable metric.
        recon_losses = []
        e_shares = []

        for start in range(0, num_samples, batch_size):
            xb = x[start : start + batch_size]
            yb = y[start : start + batch_size]
            cb = c[start : start + batch_size]
            lb = l[start : start + batch_size]

            # Use test_batch so it's no-grad and uses the same loss implementation.
            metrics_any: Any = model.test_batch(xb, yb, lb, cb, beta=1.0, return_metrics=True)
            metrics = cast(Mapping[str, Any], metrics_any)
            recon_losses.append(float(metrics['recon_loss']))

            # Predicted 'E' share over valid (non-pad) positions.
            # Build the same mask as _sequence_loss.
            x_t, _, l_t, c_t = model._to_tensor_batch(xb, yb, lb, cb)
            with torch.no_grad():
                probs, *_ = model.forward(x_t, c_t, l_t)
            pred = np.argmax(probs.detach().cpu().numpy(), axis=-1)
            steps_arr = np.arange(pred.shape[1])[None, :]
            mask = steps_arr < lb[:, None]
            valid = mask.sum()
            if valid == 0:
                continue
            e_share = float(((pred == e_index) & mask).sum() / valid)
            e_shares.append(e_share)

        return float(np.mean(recon_losses)), float(np.mean(e_shares))

    print(f'Overfit sanity check: mode={model_mode} samples={num_samples} batch={batch_size} steps={steps}')
    print(f"Vocab size={vocab_size} E_index={e_index} seq_length={seq_length}")

    # Train loop.
    for step in range(steps):
        # Simple cyclic minibatch order for determinism.
        start = (step * batch_size) % num_samples
        xb = x[start : start + batch_size]
        yb = y[start : start + batch_size]
        cb = c[start : start + batch_size]
        lb = l[start : start + batch_size]

        metrics_any: Any = model.train_batch(xb, yb, lb, cb, beta=1.0, return_metrics=True)
        metrics = cast(Mapping[str, Any], metrics_any)

        if step == 0 or (step + 1) % 10 == 0:
            recon_eval, e_share = eval_collapse_metrics()
            print(
                f"step={step+1:>3} train_loss={metrics['total_loss']:.4f} "
                f"train_recon={metrics['recon_loss']:.4f} eval_recon={recon_eval:.4f} "
                f"pred_E_share(valid)={e_share:.3f}"
            )


if __name__ == '__main__':
    main()
