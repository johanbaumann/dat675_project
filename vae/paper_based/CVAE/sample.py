
from __future__ import annotations

import itertools
import os
import time as t
from typing import Callable, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.rdMolDescriptors import CalcTPSA

from model import CVAE
from utils import (
    collect_new_unique_from_raw,
    compose_train_config_from_dict,
    convert_to_smiles,
    infer_training_config_path,
    load_sampling_metadata,
    load_checkpoint_model_config,
    load_json,
    resolve_checkpoint_path,
    save_pickle_gz,
    load_training_canonical_smiles,
)


QUALITY_METRIC_LABELS = {
    'validity': 'Validity',
    'uniqueness': 'Uniqueness',
    'novelty': 'Novelty',
    'acceptance_rate': 'Acceptance',
}


def _sample_batch_strings(
    *,
    model: CVAE,
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
) -> list[str]:
    """Sample one batch and return decoded strings (still containing 'E' padding)."""
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
    return [convert_to_smiles(generated[i], charset) for i in range(len(generated))]


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
    """Print quality metrics with human-readable labels."""
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

    # Compute V/U/N and any extra metrics before printing summary.
    metrics = _compute_quality_metrics(stats, extra_metric_fns=extra_metric_fns)

    print(80*'=')
    if scope_label:
        print(scope_label)
    print('V.U.N quality metrics:')
    _print_metric_lines(metrics, metric_labels=metric_labels)

    print(80*'=')
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
    print(80*'=')
    print('\n \n')


def _default_pickle_output_path(result_filename: str) -> str:
    """Build default compressed molecule filename next to CSV output."""
    root, _ = os.path.splitext(str(result_filename))
    return f'{root}.pckl.gz'


def _default_quality_summary_output_path(result_filename: str) -> str:
    """Build default quality summary filename next to CSV output."""
    root, _ = os.path.splitext(str(result_filename))
    return f'{root}_quality_summary.csv'


def _build_quality_summary_row(
    *,
    stats: dict,
    run_scope: str,
    num_molecules_saved: int,
    config: dict,
) -> dict:
    """Build one-row summary with V/U/N and detailed rejection breakdown."""
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
    return {
        'run_scope': str(run_scope),
        'run_property_sweep': bool(config.get('run_property_sweep', False)),
        'num_molecules_saved': int(num_molecules_saved),
        'num_unique_requested': int(config.get('num_unique')) if config.get('num_unique') is not None else np.nan,
        'max_batches': int(config.get('max_batches')) if config.get('max_batches') is not None else np.nan,
        'do_sample': bool(config.get('do_sample', True)),
        'temperature': float(config.get('temperature', 1.0)),
        'top_k': int(config.get('top_k')) if config.get('top_k') is not None else np.nan,
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
    """Persist one-row quality summary CSV and return its path."""
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
    mw_tol = config.get('mw_tolerance')
    logp_tol = config.get('logp_tolerance')
    min_tpsa = config.get('min_tpsa')
    max_heavy_atoms = config.get('max_heavy_atoms')
    max_canonical_len = config.get('max_canonical_smiles_len')

    # If no constraints set, return None (no extra filtering).
    if all(v is None for v in [mw_tol, logp_tol, min_tpsa, max_heavy_atoms, max_canonical_len]):
        return None

    mw_target = float(target_row[0])
    logp_target = float(target_row[1])

    def predicate(can: str, mol: Chem.Mol) -> bool:
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


def generate_unique_molecules(
    *,
    model: CVAE,
    charset: np.ndarray,
    target_prop: np.ndarray,
    start_codon: np.ndarray,
    seq_length: int,
    num_unique: int,
    max_batches: Optional[int],
    mean: float,
    stddev: float,
    batch_size: int,
    latent_size: int,
    training_smiles: set,
    do_sample: bool,
    temperature: float,
    top_k: Optional[int],
    strip_salts: bool,
    decharge: bool,
    canonicalize_tautomer: bool,
    accept_predicate: Optional[Callable[[str, Chem.Mol], bool]] = None,
) -> tuple[list[Chem.Mol], list[str], dict]:
    """Generate molecules until `num_unique` unique valid molecules are collected."""
    if num_unique <= 0:
        raise ValueError('num_unique must be a positive integer')

    unique_mols_by_smiles: dict[str, Chem.Mol] = {}
    seen_smiles: set = set()
    batches = 0
    total_trials = 0
    total_stats = _new_stats()

    while len(unique_mols_by_smiles) < num_unique:
        if max_batches is not None and batches >= max_batches:
            print(
                f"Reached max_batches={max_batches} with {len(unique_mols_by_smiles)}/{num_unique} unique molecules."
            )
            break

        raw = _sample_batch_strings(
            model=model,
            charset=charset,
            target_prop=target_prop,
            start_codon=start_codon,
            seq_length=seq_length,
            mean=mean,
            stddev=stddev,
            batch_size=batch_size,
            latent_size=latent_size,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
        )
        total_trials += len(raw)
        batches += 1

        accepted, batch_stats = collect_new_unique_from_raw(
            raw_strings=raw,
            seen_smiles=seen_smiles,
            training_smiles=training_smiles,
            eos_token='E',
            accept_predicate=accept_predicate,
            strip_salts=strip_salts,
            decharge=decharge,
            canonicalize_tautomer=canonicalize_tautomer,
        )
        _accumulate_stats(total_stats, batch_stats)

        for can, mol in accepted:
            unique_mols_by_smiles[can] = mol
            if len(unique_mols_by_smiles) >= num_unique:
                break

        if batches == 1 or batches % 10 == 0:
            print(
                f"Progress: {len(unique_mols_by_smiles)}/{num_unique} unique molecules "
                f"after {batches} batches ({total_trials} trials)"
            )
            print(
                f"  batch quality -> accepted: {batch_stats['accepted']}, invalid_or_empty: {batch_stats['invalid_or_empty']}, "
                f"in_training: {batch_stats['in_training']}, duplicate: {batch_stats['duplicate']}, rejected_by_filter: {batch_stats.get('rejected_by_filter', 0)}, "
                f"salt_stripped: {batch_stats.get('salt_stripped', 0)}, tautomer_canonicalized: {batch_stats.get('tautomer_canonicalized', 0)}"
            )

    _print_quality_stats(total_stats)

    smiles_out = sorted(unique_mols_by_smiles.keys())[:num_unique]
    mols_out = [unique_mols_by_smiles[s] for s in smiles_out]
    return mols_out, smiles_out, total_stats


def generate_fixed_iterations(
    *,
    model: CVAE,
    charset: np.ndarray,
    target_prop: np.ndarray,
    start_codon: np.ndarray,
    seq_length: int,
    num_iteration: int,
    mean: float,
    stddev: float,
    batch_size: int,
    latent_size: int,
    training_smiles: set,
    do_sample: bool,
    temperature: float,
    top_k: Optional[int],
    strip_salts: bool,
    decharge: bool,
    canonicalize_tautomer: bool,
    accept_predicate: Optional[Callable[[str, Chem.Mol], bool]] = None,
) -> tuple[list[Chem.Mol], list[str], dict]:
    """Old behavior: run for a fixed number of iterations, then deduplicate."""
    unique_mols_by_smiles: dict[str, Chem.Mol] = {}
    seen_smiles: set = set()
    total_trials = 0
    total_stats = _new_stats()

    for _ in range(num_iteration):
        raw = _sample_batch_strings(
            model=model,
            charset=charset,
            target_prop=target_prop,
            start_codon=start_codon,
            seq_length=seq_length,
            mean=mean,
            stddev=stddev,
            batch_size=batch_size,
            latent_size=latent_size,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
        )
        total_trials += len(raw)
        accepted, batch_stats = collect_new_unique_from_raw(
            raw_strings=raw,
            seen_smiles=seen_smiles,
            training_smiles=training_smiles,
            eos_token='E',
            accept_predicate=accept_predicate,
            strip_salts=strip_salts,
            decharge=decharge,
            canonicalize_tautomer=canonicalize_tautomer,
        )
        _accumulate_stats(total_stats, batch_stats)

        for can, mol in accepted:
            unique_mols_by_smiles.setdefault(can, mol)

    print(f'number of generated trials : {total_trials}')
    print(f'number of unique valid molecules : {len(unique_mols_by_smiles)}')
    _print_quality_stats(total_stats)
    smiles_out = sorted(unique_mols_by_smiles.keys())
    mols_out = [unique_mols_by_smiles[s] for s in smiles_out]
    return mols_out, smiles_out, total_stats

def generate_unique_over_param_sweeps(
    *, # for better readability....
    props: dict[str, list[float]]|dict[str, np.ndarray|list[float]], 
    model: CVAE,
    model_conf: dict,
    config: dict,
    charset: np.ndarray,
    start_codon: np.ndarray,
    training_smiles: set,
    top_k: Optional[int],
    unique: int,
) -> pd.DataFrame:
    """
    Sweep over a range of for example MW and LogP targets.

    This is since the synthetic training data that one will use should be -
    representative of the whole property space, not just a narrow slice.

    args:
        - props: a dict of the property name (not shure if the name is used, but will keep for debugging..)
        and a list of target values to sweep over for that property. For example:
        {
            'MW': [200.0, 300.0, 400.0],
            'LogP': [1.0, 3.0, 5.0],
        }
        props can also be a dict of:
        {
            'MW': np.ndarray of shape (num_targets,), (if one use a linspace or something instead of a list)
            'LogP': np.ndarray of shape (num_targets,),
        }
        where the i-th element of each array corresponds to a target property combination. The function will iterate over the cartesian product of the property values, so if we have 3 MW targets and 3 LogP targets, we will generate for all 9 combinations of (MW, LogP).
        - mopdel: the trained CVAE model to sample from
        - model_conf: the model configuration dict (loaded from training config or checkpoint metadata)
        - config: the runtime sampling configuration dict (defined in main())
        - charset: the array of characters used for decoding model outputs
        - start_codon: the array representing the start token for sampling
        - training_smiles: a set of canonical SMILES from the training data, used for novelty filtering
        - top_k: the top_k sampling parameter to control diversity/quality of outputs
        - unique: number of unique molecules to generate for each target property combination


    
    
    """
    results = []
    prop_names = list(props.keys())
    prop_values = list(props.values())
    # will iterate over the cartesian product of the property values, -
    # so if we have 3 MW targets and 3 LogP targets, we will generate for all 9 combinations of (MW, LogP).
    for i, target_row in enumerate(itertools.product(*prop_values)):
        
        
        
        print(80*'=')
        
        print(f"Generating for target properties: {dict(zip(prop_names, target_row))}")
        
        print(f"Combo: ({i+1}/{np.prod([len(v) for v in prop_values])})")

        print(80*'=')


        target_prop = np.array(
            [list(target_row) for _ in range(int(model_conf['batch_size']))],
            dtype=np.float32,
        )
        target_prop = normalize_like_training(
            target_prop,
            model_conf.get('prop_norm_mean'),
            model_conf.get('prop_norm_std'),
        )
        accept_predicate = _build_accept_predicate(config=config, target_row=list(target_row))
        ms, smiles, stats = generate_unique_molecules(
            model=model,
            charset=charset,
            target_prop=target_prop,
            start_codon=start_codon,
            seq_length=int(model_conf['seq_length']),
            num_unique=unique,
            max_batches=config['max_batches'],
            mean=float(model_conf['mean']),
            stddev=float(model_conf['stddev']),
            batch_size=int(model_conf['batch_size']),
            latent_size=int(model_conf['latent_size']),
            training_smiles=training_smiles,
            do_sample=bool(config.get('do_sample', True)),
            temperature=float(config.get('temperature', 1.0)),
            top_k=top_k,
            strip_salts=bool(config.get('strip_salts', True)),
            decharge=bool(config.get('decharge', True)),
            canonicalize_tautomer=bool(config.get('canonicalize_tautomer', False)),
            accept_predicate=accept_predicate,
        )
        metrics = _compute_quality_metrics(stats)

        row = {
            'target_properties': tuple(float(v) for v in target_row),
            'num_molecules': len(ms),
            'molecules': ms,
            'smiles': smiles,
            'total_generated': int(stats['total_generated']),
            'accepted': int(stats['accepted']),
            'invalid_or_empty': int(stats['invalid_or_empty']),
            'in_training': int(stats['in_training']),
            'duplicate': int(stats['duplicate']),
            'rejected_by_filter': int(stats.get('rejected_by_filter', 0)),
            'salt_stripped': int(stats.get('salt_stripped', 0)),
            'tautomer_canonicalized': int(stats.get('tautomer_canonicalized', 0)),
            'validity': float(metrics['validity']),
            'uniqueness': float(metrics['uniqueness']),
            'novelty': float(metrics['novelty']),
            'acceptance_rate': float(metrics['acceptance_rate']),
        }
        for prop_name, prop_value in zip(prop_names, target_row):
            row[str(prop_name)] = float(prop_value)
        results.append(row)
    return pd.DataFrame(results)

def converts_sweep_to_list(df:pd.DataFrame) -> tuple[list[Chem.Mol],list[str]]:
    all_mols = []
    all_smiles = []
    for _, row in df.iterrows():
        all_mols.extend(row['molecules'])
        all_smiles.extend(row['smiles'])
    return all_mols, all_smiles

def create_and_restore_model(config: dict, model_config: dict,vocab_size:int) -> CVAE:
    model = CVAE(vocab_size=vocab_size, args=model_config)
    model.restore(config['save_file'])
    print('Number of parameters : ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model

def normalize_like_training(
    target_prop: np.ndarray,
    prop_norm_mean: Optional[list],
    prop_norm_std: Optional[list],
) -> np.ndarray:
    """Apply the same normalization used during training to the target properties."""

    if prop_norm_mean is None or prop_norm_std is None:
        return target_prop

    mean_arr = np.array(prop_norm_mean, dtype=np.float32)
    std_arr = np.array(prop_norm_std, dtype=np.float32)
    std_arr = np.where(std_arr < 1e-8, 1.0, std_arr)
    return (target_prop - mean_arr) / std_arr


def compose_runtime_sample_config(runtime_config: dict) -> dict:
    """Compose nested runtime config into a flat dict used by sampling code."""
    model = runtime_config.get('model', {})
    generation = runtime_config.get('generation', {})
    sampling = runtime_config.get('sampling', {})
    filters = runtime_config.get('filters', {})
    cleanup = runtime_config.get('cleanup', {})
    output = runtime_config.get('output', {})
    sweep = runtime_config.get('sweep', {})

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
        # Standardization controls for filtering/novelty checks.
        # Tautomer canonicalization is optional because it can be slow.
        # Keep fallback to filters for backward compatibility with older configs.
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
    }






# =========================================================================
# NOTE: Under this is the setup config:
## =========================================================================





if __name__ == '__main__':

    # Silence RDKit parse error spam (we still count invalid SMILES).
    RDLogger.DisableLog('rdApp.error')
    RDLogger.DisableLog('rdApp.warning')

    start = t.time()

    # Runtime sampling options.
    # Model architecture/training hyperparameters are loaded from training_config.json.

    model_type = 'lstm' 

    


    runtime_config = {
        'model': {
            # Prefer selecting checkpoint from a run folder created by train.py.
            # If save_file is provided, it takes precedence over run_dir.
            'save_file': None,
            #'run_dir': 'save/huge_generation_lstm',
            'run_dir': 'save/run_20260219_230438',
            'checkpoint_glob': 'model_best.ckpt-*.pt',
            'training_config_file': None, # If None, will try to infer from checkpoint metadata or filename patterns.
        },
        'generation': {
            'batch_size': 64,  # Paper used 256, but that may cause OOM on smaller GPUs.
            'num_iteration': 10,  # Number of batches to sample (legacy fixed-iteration mode).
            'num_unique': 3_000,  # 30k unique molecules for each sweep point.
            'max_batches': 5000,
            'target_prop': '300.0 3.0',
            'prop_file': None,
            'seq_length': None,
            'mean': None,
            'stddev': None,
        },
        'sampling': {
            # Sampling controls. Greedy decoding (do_sample=False) 
            'do_sample': False,
            # Sweep for this checkpoint suggests ~temperature=0.6, top_k=20 gives much higher unique+novel acceptance.
            'temperature': 0.6, # higher temperature -> more random, lower temperature -> more valid, less diverse
            'top_k': 20, # limits sampling to the top_k most probable tokens at each step. Can help improve validity at low temperatures.
        },
        'filters': {
            # Optional constraints to keep outputs close to target properties.
            # These values assume target_prop is MW then LogP.
            # If you set them to None, sampling accepts any valid/novel molecule.
            'mw_tolerance': 200.0,
            'logp_tolerance': 5.0,
            # Optional: enforce polarity so TPSA isn't ~0.0 for hydrocarbon-only molecules.
            'min_tpsa': None,
            # Hard caps (optional) to prevent very large molecules due to halogen-heavy strings.
            'max_heavy_atoms': 60,
            # Canonical SMILES can be longer than seq_length because RDKit may insert brackets.
            'max_canonical_smiles_len': None,
            # If True, molecules already present in training/property file are rejected.
            'exclude_training': True,
        },
        'cleanup': {
            # Canonicalization controls used for duplicate/novelty checks.
            # Fast default: parse + canonical SMILES + decharge; tautomer optional.
            'strip_salts': True,
            'decharge': True,
            'canonicalize_tautomer': False,
        },
        'sweep': {
            'enabled': True,
            'prop_profile': {
                'MW': np.linspace(150.0, 500.0, num=10),
                'LogP': np.linspace(-4.0, 6.0, num=10),
            },
        },
        'output': {
            #'result_filename': 'CVAE_lstm_300k_test.txt',
            'result_filename': 'CVAE_transformer_300k_test.txt',
            # If None, defaults to result filename stem + '.pckl.gz'.
            'molecules_pickle_filename': None,
            # If None, defaults to result filename stem + '_quality_summary.csv'.
            'quality_summary_filename': None,
            #'sweep_stats_filename': 'CVAE_lstm_300k_test.csv',
            'sweep_stats_filename': 'CVAE_transformer_300k_test.csv',
            
        },
    }

    config = compose_runtime_sample_config(runtime_config)
    config['save_file'] = resolve_checkpoint_path(
        save_file=config.get('save_file'),
        run_dir=config.get('run_dir'),
        checkpoint_glob=str(config.get('checkpoint_glob', 'model_best.ckpt-*.pt')),
    )
    print(f"resolved checkpoint: {config['save_file']}")

    training_config_path = config['training_config_file']

    # Prefer checkpoint-embedded model_config (most reliable).
    model_config = load_checkpoint_model_config(config['save_file'])
    if model_config is not None:
        print('loaded model config from checkpoint metadata')
    else:
        if training_config_path is None:
            training_config_path = infer_training_config_path(config['save_file'])
        training_config = load_json(training_config_path)
        print(f'loaded training config from: {training_config_path}')
        # Support both grouped and legacy flat training config formats.
        model_config = compose_train_config_from_dict(training_config)

    # Allow a few runtime overrides when needed (batch_size/seq_length/mean/stddev/prop_file).
    for key in ['batch_size', 'prop_file', 'seq_length', 'mean', 'stddev']:
        if config.get(key) is not None:
            model_config[key] = config[key]

    # Build vocabulary/charset and infer num_prop from property file (fast metadata path).
    charset, vocab, inferred_num_prop = load_sampling_metadata(
        model_config['prop_file'],
        int(model_config['seq_length']),
    )
    vocab_size = len(charset)
    trained_num_prop = int(model_config.get('num_prop', inferred_num_prop))
    if trained_num_prop != inferred_num_prop:
        raise ValueError(
            f"Mismatch between training config num_prop ({trained_num_prop}) and "
            f"property file columns ({inferred_num_prop})."
        )
    model_config['num_prop'] = inferred_num_prop

    # Canonical training-set SMILES used for novelty filtering.
    if bool(config.get('exclude_training', True)):
        training_smiles = load_training_canonical_smiles(
            model_config['prop_file'],
            int(model_config['seq_length']),
            strip_salts=bool(config.get('strip_salts', True)),
            decharge=bool(config.get('decharge', True)),
            canonicalize_tautomer=bool(config.get('canonicalize_tautomer', False)),
        )
        print(f'training molecules available for exclusion: {len(training_smiles)}')
    else:
        training_smiles = set()
        print('training-set exclusion is disabled (exclude_training=False)')

    # Create and restore model.
    model = create_and_restore_model(config, model_config, vocab_size)

    # Target property conditioning: replicate the target row for the whole batch.
    try:
        target_row = [float(p) for p in str(config['target_prop']).split()]
    except Exception:
        raise ValueError(
            'target_prop should be a string of space separated values. '
            'e.g. "300.0 3.0" for two properties (MW, LogP)'
        )

    if len(target_row) != int(model_config['num_prop']):
        raise ValueError(
            f"target_prop has {len(target_row)} values, but model expects "
            f"{int(model_config['num_prop'])} properties."
        )

    target_prop = np.array([target_row for _ in range(int(model_config['batch_size']))], dtype=np.float32)

    # Build additional acceptance filter based on *original* (unnormalized) target_row.
    accept_predicate = _build_accept_predicate(config=config, target_row=target_row)
    if accept_predicate is None:
        print('acceptance constraints: disabled (no extra filtering)')
    else:
        print(
            'acceptance constraints: '
            f"mw_tol={config.get('mw_tolerance')} logp_tol={config.get('logp_tolerance')} "
            f"min_tpsa={config.get('min_tpsa')} max_heavy_atoms={config.get('max_heavy_atoms')} "
            f"max_canonical_smiles_len={config.get('max_canonical_smiles_len')}"
        )
    print(
        'canonicalization options: '
        f"strip_salts={bool(config.get('strip_salts', True))} "
        f"decharge={bool(config.get('decharge', True))} "
        f"canonicalize_tautomer={bool(config.get('canonicalize_tautomer', False))}"
    )

    # If training used standardized properties, apply the same transform here.
    prop_norm_mean = model_config.get('prop_norm_mean')
    prop_norm_std = model_config.get('prop_norm_std')
    target_prop = normalize_like_training(target_prop, prop_norm_mean, prop_norm_std)

    # Start token: 'X'. In this dataset, 'X' is appended to the vocab in `load_data()`.
    start_codon = np.array([np.array([vocab['X']]) for _ in range(int(model_config['batch_size']))])

    top_k_val = config.get('top_k')
    top_k = (None if top_k_val is None else int(top_k_val))








    run_scope = 'single_target'
    run_stats = _new_stats()

    if bool(config.get('run_property_sweep', False)):
        # Sweep ranges are defined at the top in runtime_config['sweep']['prop_profile'].
        props_to_sweep = config['prop_profile']
        sweep_results = generate_unique_over_param_sweeps(
            props=props_to_sweep,
            model=model,
            model_conf=model_config,
            config=config,
            charset=charset,
            start_codon=start_codon,
            training_smiles=training_smiles,
            top_k=top_k,
            unique=int(config['num_unique']),
        )
        ms, smiles = converts_sweep_to_list(sweep_results)

        heatmap_cols = [
            col for col in ['MW', 'LogP', 'validity', 'uniqueness', 'novelty', 'acceptance_rate',
                            'total_generated', 'accepted', 'invalid_or_empty', 'in_training',
                            'duplicate', 'rejected_by_filter', 'salt_stripped', 'tautomer_canonicalized']
            if col in sweep_results.columns
        ]
        sweep_results.loc[:, heatmap_cols].to_csv(config['sweep_stats_filename'], index=False)
        print(f"saved sweep statistics: {config['sweep_stats_filename']}")

        sweep_total_stats = _aggregate_stats_from_sweep_results(sweep_results)
        run_scope = 'sweep_all_pairs'
        run_stats = sweep_total_stats
        _print_quality_stats(
            sweep_total_stats,
            scope_label='WHOLE GENERATED SWEEP (all property pairs combined)',
        )
    elif config['num_unique'] is not None:
        ms, smiles, run_stats = generate_unique_molecules(
            model=model,
            charset=charset,
            target_prop=target_prop,
            start_codon=start_codon,
            seq_length=int(model_config['seq_length']),
            num_unique=int(config['num_unique']),
            max_batches=config['max_batches'],
            mean=float(model_config['mean']),
            stddev=float(model_config['stddev']),
            batch_size=int(model_config['batch_size']),
            latent_size=int(model_config['latent_size']),
            training_smiles=training_smiles,
            do_sample=bool(config.get('do_sample', True)),
            temperature=float(config.get('temperature', 1.0)),
            top_k=top_k,
            strip_salts=bool(config.get('strip_salts', True)),
            decharge=bool(config.get('decharge', True)),
            canonicalize_tautomer=bool(config.get('canonicalize_tautomer', False)),
            accept_predicate=accept_predicate,
        )
    else:
        ms, smiles, run_stats = generate_fixed_iterations(
            model=model,
            charset=charset,
            target_prop=target_prop,
            start_codon=start_codon,
            seq_length=int(model_config['seq_length']),
            num_iteration=int(config['num_iteration']),
            mean=float(model_config['mean']),
            stddev=float(model_config['stddev']),
            batch_size=int(model_config['batch_size']),
            latent_size=int(model_config['latent_size']),
            training_smiles=training_smiles,
            do_sample=bool(config.get('do_sample', True)),
            temperature=float(config.get('temperature', 1.0)),
            top_k=top_k,
            strip_salts=bool(config.get('strip_salts', True)),
            decharge=bool(config.get('decharge', True)),
            canonicalize_tautomer=bool(config.get('canonicalize_tautomer', False)),
            accept_predicate=accept_predicate,
        )

    print('number of valid smiles : ', len(ms))

    # Save generated molecules in compressed binary form to reduce output size.
    molecules_pickle_filename = config.get('molecules_pickle_filename')
    if molecules_pickle_filename is None:
        # do not save!
    
        #molecules_pickle_filename = _default_pickle_output_path(config['result_filename'])
        print('no pickle path, not saving compressed mols')
    if molecules_pickle_filename is not None:
        save_pickle_gz(
            molecules_pickle_filename,
            {
                'smiles': smiles,
                'molecules': ms,
                'num_molecules': len(ms),
                'saved_at_unix': t.time(),
            },
        )
        print(f'saved compressed molecules: {molecules_pickle_filename}')

    quality_summary_filename = _save_quality_summary_csv(
        stats=run_stats,
        run_scope=run_scope,
        num_molecules_saved=len(ms),
        config=config,
    )
    print(f'saved quality summary: {quality_summary_filename}')

    # Compute properties and write results.
    if len(ms) == 0:
        df = pd.DataFrame({'smiles': [], 'MW': [], 'LogP': [], 'TPSA': []})
    else:
        mw = [ExactMolWt(m) for m in ms]
        logp = [MolLogP(m) for m in ms]
        tpsa = [CalcTPSA(m) for m in ms]
        df = pd.DataFrame({'smiles': smiles, 'MW': mw, 'LogP': logp, 'TPSA': tpsa})

    print(df.describe())
    df.to_csv(config['result_filename'], index=False)

    end_time = t.time()
    print(f'time to run: {end_time - start}')

