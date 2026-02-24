from __future__ import annotations

"""utils_labels.py

Helper utilities for the label-augmented CVAE training/sampling scripts.

Goal:
- Keep `train_labels.py` and `sample_labels.py` focused on their main flows.
- Centralize small reusable helpers (KL annealing, checkpoint IO helpers,
  sampling stats/quality metrics, label prediction-from-latent helpers).

Notes:
- This module intentionally uses lazy imports for RDKit/model modules so that
  importing training utilities does not require RDKit to be installed.
"""

import glob
import os
from copy import deepcopy
from typing import Callable, Optional

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Training helpers
# -----------------------------------------------------------------------------


def log_cuda_mem(prefix: str = "") -> None:
    import torch

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        print(f"{prefix} cuda_mem_allocated={alloc:.1f} MiB reserved={reserved:.1f} MiB")


def get_kl_beta(epoch: int, cfg: dict) -> float:
    """KL annealing schedule used by `train_labels.py`."""
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


def apply_training_preset(cfg: dict) -> dict:
    """Apply named training presets (kept compatible with existing configs)."""
    preset = str(cfg.get('training_preset', 'custom')).strip().lower()
    if preset in ('', 'custom', 'none'):
        print('training preset: custom (no automatic overrides)')
        return cfg

    if preset == 'stable_transformer':
        cfg.update(
            {
                'model_mode': 'transformer',
                'optimizer': 'adamw',
                'weight_decay': 0.001,
                'use_amp': True,
                'kl_anneal_enabled': True,
                'kl_anneal_start_beta': 0.01,
                'kl_anneal_max_beta': 1.0,
                'kl_anneal_hold_epochs': 0,
                'kl_anneal_warmup_epochs': 8,
                'diagnostics_every': 1,
            }
        )
        print('training preset: stable_transformer (applied)')
        return cfg

    raise ValueError("training_preset must be one of: 'custom', 'stable_transformer'")


def save_history_csv(*, config: dict, history: dict) -> None:
    history_df = pd.DataFrame(history)
    history_df.to_csv(config['save_dir'] + '/history.csv', index=False)


def save_current_checkpoint(*, epoch: int, config: dict, model, model_config: dict, suffix: str = "") -> None:
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
    model,
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


# -----------------------------------------------------------------------------
# Sampling + quality metrics helpers
# -----------------------------------------------------------------------------


def _default_prop_names(num_prop: int) -> list[str]:
    if int(num_prop) == 2:
        return ['MW', 'LogP']
    return [f'prop_{i}' for i in range(int(num_prop))]


def _compute_rdkit_descriptors(mols) -> dict[str, list[float]]:
    """Compute a small set of RDKit descriptors for already-validated molecules."""
    # Lazy import so `train_labels.py` can import this module without RDKit.
    from rdkit.Chem.Crippen import MolLogP  # type: ignore[import]
    from rdkit.Chem.Descriptors import ExactMolWt  # type: ignore[import]
    from rdkit.Chem.rdMolDescriptors import CalcTPSA  # type: ignore[import]

    mw: list[float] = []
    logp: list[float] = []
    tpsa: list[float] = []
    for m in mols:
        try:
            mw.append(float(ExactMolWt(m)))
        except Exception:
            mw.append(np.nan)
        try:
            logp.append(float(MolLogP(m)))
        except Exception:
            logp.append(np.nan)
        try:
            tpsa.append(float(CalcTPSA(m)))
        except Exception:
            tpsa.append(np.nan)
    return {'MW': mw, 'LogP': logp, 'TPSA': tpsa}


def _model_supports_label_prediction(model) -> bool:
    return bool(getattr(model, 'predict_labels', False)) and hasattr(model, 'predict_label')


def _predict_labels_from_latent(
    *,
    model,
    latent_vector: np.ndarray,
    condition: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Predict labels from a batch of sampled latent vectors."""
    if not _model_supports_label_prediction(model):
        return None

    import torch

    z = torch.as_tensor(latent_vector, dtype=torch.float32, device=model.device)
    c = None
    if bool(getattr(model, 'include_condition_in_label_head', False)):
        if condition is None:
            raise ValueError(
                'Loaded model requires conditioning for label prediction '
                '(include_condition_in_label_head=True), but no condition was provided.'
            )
        c = torch.as_tensor(condition, dtype=torch.float32, device=model.device)
    with torch.no_grad():
        y_hat = model.predict_label(z, c=c)
    return y_hat.detach().cpu().numpy().astype(np.float32)


QUALITY_METRIC_LABELS = {
    'validity': 'Validity',
    'uniqueness': 'Uniqueness',
    'novelty': 'Novelty',
    'acceptance_rate': 'Acceptance',
}


def _sample_batch_strings(
    *,
    model,
    charset: np.ndarray,
    target_prop: np.ndarray,
    start_codon: np.ndarray,
    seq_length: int,
    mean: float,
    stddev: float,
    batch_size: int,
    latent_size: int,
    do_sample: bool,
    temperature: float,
    top_k: Optional[int],
) -> tuple[list[str], np.ndarray, Optional[np.ndarray]]:
    from utils import convert_to_smiles

    latent_vector = np.random.normal(mean, stddev, (batch_size, latent_size))
    generated = model.sample(
        latent_vector,
        target_prop,
        start_codon,
        seq_length,
        do_sample=bool(do_sample),
        temperature=float(temperature),
        top_k=top_k,
    )
    raw_strings = [convert_to_smiles(generated[i], charset) for i in range(len(generated))]
    pred_labels = _predict_labels_from_latent(model=model, latent_vector=latent_vector, condition=target_prop)
    return raw_strings, latent_vector, pred_labels


def _collect_new_unique_from_raw_with_payload(
    *,
    raw_strings: list[str],
    payload: Optional[np.ndarray],
    seen_smiles: set,
    training_smiles: Optional[set] = None,
    eos_token: str = 'E',
    accept_predicate: Optional[Callable[[str, object], bool]] = None,
    strip_salts: bool = True,
    decharge: bool = True,
    canonicalize_tautomer: bool = False,
):
    """Like `utils.collect_new_unique_from_raw()`, but carries an aligned payload."""
    from utils import canonicalize_for_filtering

    accepted = []
    stats = {
        'total_generated': 0,
        'accepted': 0,
        'invalid_or_empty': 0,
        'in_training': 0,
        'duplicate': 0,
        'rejected_by_filter': 0,
        'salt_stripped': 0,
        'tautomer_canonicalized': 0,
    }

    if training_smiles is None:
        training_smiles = set()

    if payload is not None and len(payload) != len(raw_strings):
        raise ValueError('payload must be None or have same length as raw_strings')

    for i, s in enumerate(raw_strings):
        stats['total_generated'] += 1
        s = s.split(eos_token)[0].strip()
        if not s:
            stats['invalid_or_empty'] += 1
            continue

        can, mol, can_info = canonicalize_for_filtering(
            s,
            strip_salts=strip_salts,
            decharge=decharge,
            canonicalize_tautomer=canonicalize_tautomer,
        )
        if can is None or mol is None:
            stats['invalid_or_empty'] += 1
            continue

        stats['salt_stripped'] += int(can_info.get('salt_stripped', 0))
        stats['tautomer_canonicalized'] += int(can_info.get('tautomer_canonicalized', 0))

        if can in training_smiles:
            stats['in_training'] += 1
            continue

        if can in seen_smiles:
            stats['duplicate'] += 1
            continue

        if accept_predicate is not None:
            try:
                ok = bool(accept_predicate(can, mol))
            except Exception:
                ok = False
            if not ok:
                stats['rejected_by_filter'] += 1
                continue

        seen_smiles.add(can)
        y_row = None
        if payload is not None:
            y_row = payload[i]
        accepted.append((can, mol, y_row))
        stats['accepted'] += 1

    return accepted, stats


def _new_stats() -> dict:
    return {
        'total_generated': 0,
        'accepted': 0,
        'invalid_or_empty': 0,
        'in_training': 0,
        'duplicate': 0,
        'rejected_by_filter': 0,
        'salt_stripped': 0,
        'tautomer_canonicalized': 0,
    }


def _accumulate_stats(total: dict, inc: dict) -> None:
    for key in total.keys():
        total[key] += int(inc.get(key, 0))


def _compute_quality_metrics(
    stats: dict,
    extra_metric_fns: Optional[dict[str, Callable[[dict, dict], float]]] = None,
) -> dict:
    total_generated = int(stats.get('total_generated', 0))
    invalid_or_empty = int(stats.get('invalid_or_empty', 0))
    in_training = int(stats.get('in_training', 0))
    duplicate = int(stats.get('duplicate', 0))
    accepted = int(stats.get('accepted', 0))
    rejected_by_filter = int(stats.get('rejected_by_filter', 0))
    salt_stripped = int(stats.get('salt_stripped', 0))
    tautomer_canonicalized = int(stats.get('tautomer_canonicalized', 0))

    if total_generated <= 0:
        metrics = {
            'validity': 0.0,
            'uniqueness': 0.0,
            'novelty': 0.0,
            'acceptance_rate': 0.0,
            'valid_count': 0,
            'novel_count': 0,
            'unique_count': 0,
            'rejected_by_filter': rejected_by_filter,
            'salt_stripped': salt_stripped,
            'tautomer_canonicalized': tautomer_canonicalized,
        }
        if extra_metric_fns:
            for metric_name, metric_fn in extra_metric_fns.items():
                try:
                    metrics[metric_name] = float(metric_fn(stats, metrics))
                except Exception:
                    metrics[metric_name] = np.nan
        return metrics

    valid_count = total_generated - invalid_or_empty
    unique_count = total_generated - duplicate
    novel_count = total_generated - in_training

    metrics = {
        'validity': float(valid_count) / float(total_generated),
        'uniqueness': float(unique_count) / float(total_generated),
        'novelty': float(novel_count) / float(total_generated),
        'acceptance_rate': float(accepted) / float(total_generated),
        'valid_count': int(valid_count),
        'novel_count': int(novel_count),
        'unique_count': int(unique_count),
        'rejected_by_filter': rejected_by_filter,
        'salt_stripped': salt_stripped,
        'tautomer_canonicalized': tautomer_canonicalized,
    }
    if extra_metric_fns:
        for metric_name, metric_fn in extra_metric_fns.items():
            try:
                metrics[metric_name] = float(metric_fn(stats, metrics))
            except Exception:
                metrics[metric_name] = np.nan
    return metrics


def _print_metric_lines(metrics: dict, metric_labels: Optional[dict[str, str]] = None) -> None:
    labels = dict(QUALITY_METRIC_LABELS)
    if metric_labels:
        labels.update(metric_labels)

    ordered_keys = list(QUALITY_METRIC_LABELS.keys())
    for metric_key in ordered_keys:
        if metric_key in metrics:
            print(f"{labels.get(metric_key, metric_key)}: {float(metrics[metric_key]):.2%}")

    for metric_key in metrics.keys():
        if metric_key in ordered_keys:
            continue
        metric_value = metrics[metric_key]
        if isinstance(metric_value, (int, float, np.floating)):
            if np.isnan(metric_value):
                print(f"{labels.get(metric_key, metric_key)}: nan")
            else:
                print(f"{labels.get(metric_key, metric_key)}: {float(metric_value):.4f}")


def _aggregate_stats_from_sweep_results(sweep_results: pd.DataFrame) -> dict:
    totals = _new_stats()
    if sweep_results is None or len(sweep_results) == 0:
        return totals

    for key in totals.keys():
        if key in sweep_results.columns:
            totals[key] = int(np.nansum(sweep_results[key].to_numpy(dtype=np.float64)))
    return totals


def _print_quality_stats(
    stats: dict,
    scope_label: Optional[str] = None,
    extra_metric_fns: Optional[dict[str, Callable[[dict, dict], float]]] = None,
    metric_labels: Optional[dict[str, str]] = None,
) -> None:
    total_generated = int(stats['total_generated'])
    accepted = int(stats['accepted'])
    invalid_or_empty = int(stats['invalid_or_empty'])
    in_training = int(stats['in_training'])
    duplicate = int(stats['duplicate'])

    metrics = _compute_quality_metrics(stats, extra_metric_fns=extra_metric_fns)

    print(80 * '=')
    if scope_label:
        print(scope_label)
    print('V.U.N quality metrics:')
    _print_metric_lines(metrics, metric_labels=metric_labels)

    print(80 * '=')
    print('Detailed quality stats:')
    not_ok = invalid_or_empty + in_training + duplicate
    not_ok_share = (float(not_ok) / float(total_generated)) if total_generated > 0 else 0.0
    rejected_by_filter = int(stats.get('rejected_by_filter', 0))
    salt_stripped = int(stats.get('salt_stripped', 0))
    tautomer_canonicalized = int(stats.get('tautomer_canonicalized', 0))

    print(f'total generated molecules: {total_generated}')
    print(f'accepted molecules: {accepted}')
    print(f'not ok molecules: {not_ok} ({not_ok_share:.2%})')
    print(
        f'not ok breakdown -> invalid_or_empty: {invalid_or_empty}, '
        f'in_training: {in_training}, duplicate: {duplicate}'
    )

    if rejected_by_filter:
        print(f'additional rejected_by_filter: {rejected_by_filter}')
    if salt_stripped:
        print(f'additional salt_stripped: {salt_stripped}')
    if tautomer_canonicalized:
        print(f'additional tautomer_canonicalized: {tautomer_canonicalized}')
    print(80 * '=')
    print('\n \n')


def _default_pickle_output_path(result_filename: str) -> str:
    root, _ = os.path.splitext(str(result_filename))
    return f'{root}.pckl.gz'


def _default_quality_summary_output_path(result_filename: str) -> str:
    root, _ = os.path.splitext(str(result_filename))
    return f'{root}_quality_summary.csv'


def _build_quality_summary_row(*, stats: dict, run_scope: str, num_molecules_saved: int, config: dict) -> dict:
    total_generated = int(stats.get('total_generated', 0))
    accepted = int(stats.get('accepted', 0))
    invalid_or_empty = int(stats.get('invalid_or_empty', 0))
    in_training = int(stats.get('in_training', 0))
    duplicate = int(stats.get('duplicate', 0))
    rejected_by_filter = int(stats.get('rejected_by_filter', 0))
    salt_stripped = int(stats.get('salt_stripped', 0))
    tautomer_canonicalized = int(stats.get('tautomer_canonicalized', 0))

    not_ok_count = invalid_or_empty + in_training + duplicate

    def _ratio(x: int) -> float:
        if total_generated <= 0:
            return 0.0
        return float(x) / float(total_generated)

    metrics = _compute_quality_metrics(stats)

    def _int_or_nan(x) -> float:
        if x is None:
            return float('nan')
        if isinstance(x, (bool,)):
            return float(int(x))
        if isinstance(x, (int, np.integer)):
            return float(int(x))
        if isinstance(x, (float, np.floating)):
            return float(int(x))
        if isinstance(x, str) and x.strip() != '':
            try:
                return float(int(float(x)))
            except Exception:
                return float('nan')
        return float('nan')

    return {
        'run_scope': str(run_scope),
        'run_property_sweep': bool(config.get('run_property_sweep', False)),
        'num_molecules_saved': int(num_molecules_saved),
        'num_unique_requested': _int_or_nan(config.get('num_unique')),
        'max_batches': _int_or_nan(config.get('max_batches')),
        'do_sample': bool(config.get('do_sample', True)),
        'temperature': float(config.get('temperature', 1.0)),
        'top_k': _int_or_nan(config.get('top_k')),
        'total_generated': total_generated,
        'accepted': accepted,
        'invalid_or_empty': invalid_or_empty,
        'in_training': in_training,
        'duplicate': duplicate,
        'rejected_by_filter': rejected_by_filter,
        'salt_stripped': salt_stripped,
        'tautomer_canonicalized': tautomer_canonicalized,
        'not_ok_count': int(not_ok_count),
        'validity': float(metrics['validity']),
        'uniqueness': float(metrics['uniqueness']),
        'novelty': float(metrics['novelty']),
        'acceptance_rate': float(metrics['acceptance_rate']),
        'not_ok_rate': _ratio(not_ok_count),
        'invalid_or_empty_rate': _ratio(invalid_or_empty),
        'in_training_rate': _ratio(in_training),
        'duplicate_rate': _ratio(duplicate),
        'rejected_by_filter_rate': _ratio(rejected_by_filter),
        'valid_count': int(metrics['valid_count']),
        'novel_count': int(metrics['novel_count']),
        'unique_count': int(metrics['unique_count']),
    }


def _save_quality_summary_csv(*, stats: dict, run_scope: str, num_molecules_saved: int, config: dict) -> str:
    quality_summary_filename = config.get('quality_summary_filename')
    if quality_summary_filename is None:
        quality_summary_filename = _default_quality_summary_output_path(config['result_filename'])

    summary_row = _build_quality_summary_row(
        stats=stats,
        run_scope=run_scope,
        num_molecules_saved=num_molecules_saved,
        config=config,
    )
    pd.DataFrame([summary_row]).to_csv(quality_summary_filename, index=False)
    return str(quality_summary_filename)


def _build_accept_predicate(*, config: dict, target_row: list[float]):
    """Build a per-molecule accept predicate used during sampling (filtering)."""
    # Lazy import (RDKit only needed for sampling).
    from rdkit.Chem.Crippen import MolLogP  # type: ignore[import]
    from rdkit.Chem.Descriptors import ExactMolWt  # type: ignore[import]
    from rdkit.Chem.rdMolDescriptors import CalcTPSA  # type: ignore[import]

    mw_tol = config.get('mw_tolerance')
    logp_tol = config.get('logp_tolerance')
    min_tpsa = config.get('min_tpsa')
    max_heavy_atoms = config.get('max_heavy_atoms')
    max_canonical_len = config.get('max_canonical_smiles_len')

    if all(v is None for v in [mw_tol, logp_tol, min_tpsa, max_heavy_atoms, max_canonical_len]):
        return None

    mw_target = float(target_row[0])
    logp_target = float(target_row[1])

    def predicate(can: str, mol) -> bool:
        if max_canonical_len is not None and len(can) > int(max_canonical_len):
            return False

        if max_heavy_atoms is not None and int(mol.GetNumHeavyAtoms()) > int(max_heavy_atoms):
            return False

        if mw_tol is not None:
            mw = float(ExactMolWt(mol))
            if abs(mw - mw_target) > float(mw_tol):
                return False

        if logp_tol is not None:
            lp = float(MolLogP(mol))
            if abs(lp - logp_target) > float(logp_tol):
                return False

        if min_tpsa is not None:
            tpsa = float(CalcTPSA(mol))
            if tpsa < float(min_tpsa):
                return False

        return True

    return predicate


def normalize_like_training(
    target_prop: np.ndarray,
    prop_norm_mean: Optional[list],
    prop_norm_std: Optional[list],
) -> np.ndarray:
    if prop_norm_mean is None or prop_norm_std is None:
        return target_prop

    mean_arr = np.array(prop_norm_mean, dtype=np.float32)
    std_arr = np.array(prop_norm_std, dtype=np.float32)
    std_arr = np.where(std_arr < 1e-8, 1.0, std_arr)
    return (target_prop - mean_arr) / std_arr


def sample_target_props_like_training(
    *,
    batch_size: int,
    prop_norm_mean: list,
    prop_norm_std: list,
    std_scale: float = 1.0,
    clip_n_std: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    mean_arr = np.asarray(prop_norm_mean, dtype=np.float32).reshape(1, -1)
    std_arr = np.asarray(prop_norm_std, dtype=np.float32).reshape(1, -1)
    std_arr = np.where(std_arr < 1e-8, 1.0, std_arr)

    bs = int(batch_size)
    sampled = rng.normal(loc=mean_arr, scale=(float(std_scale) * std_arr), size=(bs, int(mean_arr.shape[1]))).astype(
        np.float32
    )
    if clip_n_std is not None:
        clip = float(clip_n_std)
        lo = (mean_arr - (clip * std_arr)).astype(np.float32)
        hi = (mean_arr + (clip * std_arr)).astype(np.float32)
        sampled = np.clip(sampled, lo, hi)
    return sampled


def _collect_new_unique_from_raw_with_payloads(
    *,
    raw_strings: list[str],
    payload_a: Optional[np.ndarray],
    payload_b: Optional[np.ndarray],
    seen_smiles: set,
    training_smiles: Optional[set] = None,
    eos_token: str = 'E',
    accept_predicate: Optional[Callable[[str, object], bool]] = None,
    strip_salts: bool = True,
    decharge: bool = True,
    canonicalize_tautomer: bool = False,
):
    from utils import canonicalize_for_filtering

    accepted = []
    stats = {
        'total_generated': 0,
        'accepted': 0,
        'invalid_or_empty': 0,
        'in_training': 0,
        'duplicate': 0,
        'rejected_by_filter': 0,
        'salt_stripped': 0,
        'tautomer_canonicalized': 0,
    }

    if training_smiles is None:
        training_smiles = set()

    if payload_a is not None and len(payload_a) != len(raw_strings):
        raise ValueError('payload_a must be None or have same length as raw_strings')
    if payload_b is not None and len(payload_b) != len(raw_strings):
        raise ValueError('payload_b must be None or have same length as raw_strings')

    for i, s in enumerate(raw_strings):
        stats['total_generated'] += 1
        s = s.split(eos_token)[0].strip()
        if not s:
            stats['invalid_or_empty'] += 1
            continue

        can, mol, can_info = canonicalize_for_filtering(
            s,
            strip_salts=strip_salts,
            decharge=decharge,
            canonicalize_tautomer=canonicalize_tautomer,
        )
        if can is None or mol is None:
            stats['invalid_or_empty'] += 1
            continue

        stats['salt_stripped'] += int(can_info.get('salt_stripped', 0))
        stats['tautomer_canonicalized'] += int(can_info.get('tautomer_canonicalized', 0))

        if can in training_smiles:
            stats['in_training'] += 1
            continue

        if can in seen_smiles:
            stats['duplicate'] += 1
            continue

        if accept_predicate is not None:
            try:
                ok = bool(accept_predicate(can, mol))
            except Exception:
                ok = False
            if not ok:
                stats['rejected_by_filter'] += 1
                continue

        seen_smiles.add(can)
        a_row = None if payload_a is None else payload_a[i]
        b_row = None if payload_b is None else payload_b[i]
        accepted.append((can, mol, a_row, b_row))
        stats['accepted'] += 1

    return accepted, stats


def converts_sweep_to_list(df: pd.DataFrame):
    all_mols = []
    all_smiles = []
    all_pred = []
    have_pred = False
    for _, row in df.iterrows():
        all_mols.extend(row['molecules'])
        all_smiles.extend(row['smiles'])
        if 'pred_labels' in row and row['pred_labels'] is not None:
            have_pred = True
            all_pred.extend(row['pred_labels'])
    if not have_pred:
        return all_mols, all_smiles, None
    return all_mols, all_smiles, all_pred


def converts_sweep_to_list_with_targets(
    df: pd.DataFrame,
    *,
    prop_names: list[str],
):
    all_mols = []
    all_smiles = []
    all_pred = []
    all_targets = []
    have_pred = False

    if df is None or len(df) == 0:
        return [], [], None, None

    for _, row in df.iterrows():
        mols = list(row.get('molecules', []) or [])
        smiles = list(row.get('smiles', []) or [])
        preds = row.get('pred_labels', None)
        target_props = row.get('target_properties', None)

        if len(mols) != len(smiles):
            raise ValueError('sweep row has mismatched molecules/smiles lengths')

        if target_props is None:
            targets_for_row = [tuple(np.nan for _ in prop_names) for _ in range(len(smiles))]
        else:
            targets_for_row = [tuple(float(v) for v in target_props) for _ in range(len(smiles))]

        all_mols.extend(mols)
        all_smiles.extend(smiles)
        all_targets.extend(targets_for_row)

        if preds is not None:
            have_pred = True
            all_pred.extend(list(preds))
        else:
            all_pred.extend([None for _ in range(len(smiles))])

    if not have_pred:
        return all_mols, all_smiles, None, all_targets

    pred_out = [p for p in all_pred if p is not None]
    if len(pred_out) != len(all_smiles):
        print(
            'WARNING: predicted label list length mismatch after sweep aggregation. '
            'This usually means some sweep points were generated without a label head.'
        )
        return all_mols, all_smiles, None, all_targets

    return all_mols, all_smiles, pred_out, all_targets


def create_and_restore_model(config: dict, model_config: dict, vocab_size: int):
    """Create model instance matching checkpoint config, then restore weights."""
    from model import CVAE as CVAEBase
    from model_labels import CVAE as CVAEWithLabels

    use_label_model = bool(model_config.get('predict_labels', False))
    model_cls = CVAEWithLabels if use_label_model else CVAEBase
    model = model_cls(vocab_size=vocab_size, args=model_config)
    model.restore(config['save_file'])
    print('Number of parameters : ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model


def compose_runtime_sample_config(runtime_config: dict) -> dict:
    model = runtime_config.get('model', {})
    generation = runtime_config.get('generation', {})
    sampling = runtime_config.get('sampling', {})
    filters = runtime_config.get('filters', {})
    cleanup = runtime_config.get('cleanup', {})
    output = runtime_config.get('output', {})
    sweep = runtime_config.get('sweep', {})
    training_dist = runtime_config.get('training_dist', {})

    return {
        'save_file': model.get('save_file', None),
        'run_dir': model.get('run_dir', None),
        'checkpoint_glob': model.get('checkpoint_glob', 'model_best.ckpt-*.pt'),
        'training_config_file': model.get('training_config_file', None),
        'target_prop': generation.get('target_prop', '300.0 3.0'),
        'prop_file': generation.get('prop_file', None),
        'seq_length': generation.get('seq_length', None),
        'mean': generation.get('mean', None),
        'stddev': generation.get('stddev', None),
        'batch_size': generation.get('batch_size', 128),
        'num_iteration': generation.get('num_iteration', 10),
        'num_unique': generation.get('num_unique', 1000),
        'max_batches': generation.get('max_batches', 5000),
        'do_sample': sampling.get('do_sample', False),
        'temperature': sampling.get('temperature', 0.6),
        'top_k': sampling.get('top_k', 20),
        'mw_tolerance': filters.get('mw_tolerance', 200.0),
        'logp_tolerance': filters.get('logp_tolerance', 5.0),
        'min_tpsa': filters.get('min_tpsa', None),
        'max_heavy_atoms': filters.get('max_heavy_atoms', 60),
        'max_canonical_smiles_len': filters.get('max_canonical_smiles_len', None),
        'exclude_training': filters.get('exclude_training', True),
        'strip_salts': cleanup.get('strip_salts', filters.get('strip_salts', True)),
        'decharge': cleanup.get('decharge', filters.get('decharge', True)),
        'canonicalize_tautomer': cleanup.get(
            'canonicalize_tautomer',
            filters.get('canonicalize_tautomer', False),
        ),
        'result_filename': output.get('result_filename', 'CVAE_result.txt'),
        'molecules_pickle_filename': output.get('molecules_pickle_filename', None),
        'quality_summary_filename': output.get('quality_summary_filename', None),
        'sweep_stats_filename': output.get('sweep_stats_filename', 'CVAE_sweep_stats.csv'),
        'run_property_sweep': sweep.get('enabled', False),
        'prop_profile': sweep.get(
            'prop_profile',
            {
                'MW': np.linspace(200.0, 500.0, num=5),
                'LogP': [1.0, 3.0, 5.0],
            },
        ),
        'run_training_dist': training_dist.get('enabled', False),
        'training_dist_std_scale': training_dist.get('std_scale', 1.0),
        'training_dist_clip_n_std': training_dist.get('clip_n_std', 2.5),
        'training_dist_seed': training_dist.get('seed', None),
    }
