
from __future__ import annotations

import itertools
import os
import time as t
from typing import Callable, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from utils import (
    compose_train_config_from_dict,
    infer_training_config_path,
    load_sampling_metadata,
    load_checkpoint_model_config,
    load_json,
    resolve_checkpoint_path,
    save_pickle_gz,
    load_training_canonical_smiles,
)

from utils_labels import (
    QUALITY_METRIC_LABELS,
    _aggregate_stats_from_sweep_results,
    _build_accept_predicate,
    _build_quality_summary_row,
    _collect_new_unique_from_raw_with_payload,
    _collect_new_unique_from_raw_with_payloads,
    _compute_quality_metrics,
    _compute_rdkit_descriptors,
    _default_pickle_output_path,
    _default_prop_names,
    _default_quality_summary_output_path,
    _model_supports_label_prediction,
    _new_stats,
    _predict_labels_from_latent,
    _print_metric_lines,
    _print_quality_stats,
    _sample_batch_strings,
    _save_quality_summary_csv,
    _accumulate_stats,
    compose_runtime_sample_config,
    converts_sweep_to_list,
    converts_sweep_to_list_with_targets,
    create_and_restore_model,
    normalize_like_training,
    sample_target_props_like_training,
)


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
) -> tuple[list[Chem.Mol], list[str], dict, Optional[list[np.ndarray]]]:
    """Generate molecules until `num_unique` unique valid molecules are collected."""
    if num_unique <= 0:
        raise ValueError('num_unique must be a positive integer')

    unique_mols_by_smiles: dict[str, Chem.Mol] = {}
    unique_labels_by_smiles: dict[str, np.ndarray] = {}
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

        raw, _, pred_labels = _sample_batch_strings(
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

        # Keep predicted labels aligned with accepted molecules (if available).
        accepted, batch_stats = _collect_new_unique_from_raw_with_payload(
            raw_strings=raw,
            payload=pred_labels,
            seen_smiles=seen_smiles,
            training_smiles=training_smiles,
            eos_token='E',
            accept_predicate=accept_predicate,
            strip_salts=strip_salts,
            decharge=decharge,
            canonicalize_tautomer=canonicalize_tautomer,
        )
        _accumulate_stats(total_stats, batch_stats)

        for can, mol, y_row in accepted:
            unique_mols_by_smiles[can] = mol
            if y_row is not None:
                unique_labels_by_smiles[can] = y_row
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

    pred_out = None
    if len(unique_labels_by_smiles) > 0:
        pred_out = [unique_labels_by_smiles.get(s) for s in smiles_out]
    return mols_out, smiles_out, total_stats, pred_out


def generate_unique_molecules_from_training_dist(
    *,
    model: CVAE,
    charset: np.ndarray,
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
    prop_norm_mean: list,
    prop_norm_std: list,
    std_scale: float,
    clip_n_std: Optional[float],
    rng: Optional[np.random.Generator] = None,
) -> tuple[list[Chem.Mol], list[str], dict, Optional[list[np.ndarray]], list[tuple[float, ...]]]:
    """Generate unique molecules while sampling target properties near training data.

    This mode samples a fresh conditioning target `c_raw` each batch from an
    approximate training distribution, then normalizes it like training and
    decodes with that `c_norm`.

    Returns an extra `target_props_per_molecule` list aligned with `smiles_out`.
    """
    if num_unique <= 0:
        return [], [], _new_stats(), None, []

    if rng is None:
        rng = np.random.default_rng()

    unique_mols_by_smiles: dict[str, Chem.Mol] = {}
    unique_labels_by_smiles: dict[str, np.ndarray] = {}
    unique_targets_by_smiles: dict[str, tuple[float, ...]] = {}
    seen_smiles: set = set()
    batches = 0
    total_stats = _new_stats()

    while len(unique_mols_by_smiles) < num_unique:
        batches += 1
        if max_batches is not None and batches > int(max_batches):
            print(f'stopping early: reached max_batches={max_batches}')
            break

        # Sample per-batch conditioning targets in raw units, then normalize like training.
        target_prop_raw = sample_target_props_like_training(
            batch_size=int(batch_size),
            prop_norm_mean=list(prop_norm_mean),
            prop_norm_std=list(prop_norm_std),
            std_scale=float(std_scale),
            clip_n_std=clip_n_std,
            rng=rng,
        )
        target_prop = normalize_like_training(target_prop_raw, prop_norm_mean, prop_norm_std)

        raw, _, pred_labels = _sample_batch_strings(
            model=model,
            charset=charset,
            target_prop=target_prop,
            start_codon=start_codon,
            seq_length=int(seq_length),
            mean=float(mean),
            stddev=float(stddev),
            batch_size=int(batch_size),
            latent_size=int(latent_size),
            do_sample=bool(do_sample),
            temperature=float(temperature),
            top_k=top_k,
        )

        # Note: accept_predicate is intentionally not used here.
        # In training-dist mode the target differs per sample, and the existing
        # predicate closure is defined for a single fixed (MW, LogP, ...) row.
        accepted, inc_stats = _collect_new_unique_from_raw_with_payloads(
            raw_strings=raw,
            payload_a=pred_labels,
            payload_b=target_prop_raw,
            seen_smiles=seen_smiles,
            training_smiles=training_smiles,
            eos_token='E',
            accept_predicate=None,
            strip_salts=strip_salts,
            decharge=decharge,
            canonicalize_tautomer=canonicalize_tautomer,
        )
        _accumulate_stats(total_stats, inc_stats)
        # can is the canonical smiles string.
        for can, mol, y_pred, y_target_raw in accepted:
            if can in unique_mols_by_smiles:
                continue
            unique_mols_by_smiles[can] = mol
            if y_pred is not None:
                unique_labels_by_smiles[can] = np.asarray(y_pred, dtype=np.float32)
            if y_target_raw is not None:
                row = tuple(float(v) for v in np.asarray(y_target_raw, dtype=np.float32).reshape(-1).tolist())
                unique_targets_by_smiles[can] = row

        if batches % 10 == 0:
            print(f'batches={batches} unique={len(unique_mols_by_smiles)}/{num_unique}')

    _print_quality_stats(total_stats)

    smiles_out = sorted(unique_mols_by_smiles.keys())[:num_unique]
    mols_out = [unique_mols_by_smiles[s] for s in smiles_out]

    pred_out = None
    if len(unique_labels_by_smiles) > 0:
        pred_out = [unique_labels_by_smiles.get(s) for s in smiles_out]

    default_row = tuple(float('nan') for _ in range(int(len(prop_norm_mean))))
    target_props_per_molecule = [unique_targets_by_smiles.get(s, default_row) for s in smiles_out]
    return mols_out, smiles_out, total_stats, pred_out, target_props_per_molecule


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
) -> tuple[list[Chem.Mol], list[str], dict, Optional[list[np.ndarray]]]:
    """Old behavior: run for a fixed number of iterations, then deduplicate."""
    unique_mols_by_smiles: dict[str, Chem.Mol] = {}
    unique_labels_by_smiles: dict[str, np.ndarray] = {}
    seen_smiles: set = set()
    total_trials = 0
    total_stats = _new_stats()

    for _ in range(num_iteration):
        raw, _, pred_labels = _sample_batch_strings(
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

        accepted, batch_stats = _collect_new_unique_from_raw_with_payload(
            raw_strings=raw,
            payload=pred_labels,
            seen_smiles=seen_smiles,
            training_smiles=training_smiles,
            eos_token='E',
            accept_predicate=accept_predicate,
            strip_salts=strip_salts,
            decharge=decharge,
            canonicalize_tautomer=canonicalize_tautomer,
        )
        _accumulate_stats(total_stats, batch_stats)

        for can, mol, y_row in accepted:
            if can not in unique_mols_by_smiles:
                unique_mols_by_smiles[can] = mol
                if y_row is not None:
                    unique_labels_by_smiles[can] = y_row

    print(f'number of generated trials : {total_trials}')
    print(f'number of unique valid molecules : {len(unique_mols_by_smiles)}')
    _print_quality_stats(total_stats)
    smiles_out = sorted(unique_mols_by_smiles.keys())
    mols_out = [unique_mols_by_smiles[s] for s in smiles_out]

    pred_out = None
    if len(unique_labels_by_smiles) > 0:
        pred_out = [unique_labels_by_smiles.get(s) for s in smiles_out]
    return mols_out, smiles_out, total_stats, pred_out


def generate_fixed_iterations_from_training_dist(
    *,
    model: CVAE,
    charset: np.ndarray,
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
    prop_norm_mean: list,
    prop_norm_std: list,
    std_scale: float,
    clip_n_std: Optional[float],
    rng: Optional[np.random.Generator] = None,
) -> tuple[list[Chem.Mol], list[str], dict, Optional[list[np.ndarray]], list[tuple[float, ...]]]:
    """Fixed-iteration generation variant using training-distribution conditioning."""
    if rng is None:
        rng = np.random.default_rng()

    unique_mols_by_smiles: dict[str, Chem.Mol] = {}
    unique_labels_by_smiles: dict[str, np.ndarray] = {}
    unique_targets_by_smiles: dict[str, tuple[float, ...]] = {}
    seen_smiles: set = set()
    total_stats = _new_stats()

    for _ in range(int(num_iteration)):
        target_prop_raw = sample_target_props_like_training(
            batch_size=int(batch_size),
            prop_norm_mean=list(prop_norm_mean),
            prop_norm_std=list(prop_norm_std),
            std_scale=float(std_scale),
            clip_n_std=clip_n_std,
            rng=rng,
        )
        target_prop = normalize_like_training(target_prop_raw, prop_norm_mean, prop_norm_std)

        raw, _, pred_labels = _sample_batch_strings(
            model=model,
            charset=charset,
            target_prop=target_prop,
            start_codon=start_codon,
            seq_length=int(seq_length),
            mean=float(mean),
            stddev=float(stddev),
            batch_size=int(batch_size),
            latent_size=int(latent_size),
            do_sample=bool(do_sample),
            temperature=float(temperature),
            top_k=top_k,
        )

        accepted, inc_stats = _collect_new_unique_from_raw_with_payloads(
            raw_strings=raw,
            payload_a=pred_labels,
            payload_b=target_prop_raw,
            seen_smiles=seen_smiles,
            training_smiles=training_smiles,
            eos_token='E',
            accept_predicate=None,
            strip_salts=strip_salts,
            decharge=decharge,
            canonicalize_tautomer=canonicalize_tautomer,
        )
        _accumulate_stats(total_stats, inc_stats)

        for can, mol, y_pred, y_target_raw in accepted:
            if can in unique_mols_by_smiles:
                continue
            unique_mols_by_smiles[can] = mol
            if y_pred is not None:
                unique_labels_by_smiles[can] = np.asarray(y_pred, dtype=np.float32)
            if y_target_raw is not None:
                row = tuple(float(v) for v in np.asarray(y_target_raw, dtype=np.float32).reshape(-1).tolist())
                unique_targets_by_smiles[can] = row

    print(f'number of unique valid molecules : {len(unique_mols_by_smiles)}')
    _print_quality_stats(total_stats)

    smiles_out = sorted(unique_mols_by_smiles.keys())
    mols_out = [unique_mols_by_smiles[s] for s in smiles_out]

    pred_out = None
    if len(unique_labels_by_smiles) > 0:
        pred_out = [unique_labels_by_smiles.get(s) for s in smiles_out]

    default_row = tuple(float('nan') for _ in range(int(len(prop_norm_mean))))
    target_props_per_molecule = [unique_targets_by_smiles.get(s, default_row) for s in smiles_out]
    return mols_out, smiles_out, total_stats, pred_out, target_props_per_molecule

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
        ms, smiles, stats, pred_labels = generate_unique_molecules(
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
            'pred_labels': pred_labels,
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

    model_type = 'transformer' 

    


    runtime_config = {
        'model': {
            # Prefer selecting checkpoint from a run folder created by train.py.
            # If save_file is provided, it takes precedence over run_dir.
            'save_file': None,
            #'run_dir': 'save/huge_generation_lstm',
            #'run_dir': 'save/run_20260223_131822',
            #'run_dir': 'save/run_20260224_112850',
            #'run_dir': 'save/run_20260224_160237',
            'run_dir': 'save/run_20260224_205844',
            'checkpoint_glob': 'model_best.ckpt-*.pt',
            'training_config_file': None, # If None, will try to infer from checkpoint metadata or filename patterns.
        },
        'generation': {
            'batch_size': 128,  # Paper used 256, but that may cause OOM on smaller GPUs.
            'num_iteration': 10,  # Number of batches to sample (legacy fixed-iteration mode).
            'num_unique': 300_000,#3_000,  # 30k unique molecules for each sweep point.
            'max_batches': 5000,
            'target_prop': '300.0 3.0',
            'prop_file': None,
            'seq_length': None,
            'mean': None,
            'stddev': None,
        },
        'sampling': {
            # Sampling controls. Greedy decoding (do_sample=False) 
            'do_sample': True,
            # Sweep for this checkpoint suggests ~temperature=0.6, top_k=20 gives much higher unique+novel acceptance.
            'temperature': 0.9, # higher temperature -> more random, lower temperature -> more valid, less diverse
            'top_k': 20, # limits sampling to the top_k most probable tokens at each step. Can help improve validity at low temperatures.
        },
        'filters': {
            # Optional constraints to keep outputs close to target properties.
            # These values assume target_prop is MW then LogP.
            # If you set them to None, sampling accepts any valid/novel molecule.
            'mw_tolerance': 100.0, #200
            'logp_tolerance': 2.0, # 5.0
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
            'enabled': False,
            'prop_profile': {
                'MW': np.linspace(150.0, 500.0, num=10),
                'LogP': np.linspace(-4.0, 6.0, num=10),
            },
        },
        'training_dist': {
            # Optional 3rd conditioning mode:
            # sample target properties from an approximate training-data distribution.
            # Useful when you want to generate molecules where the label head tends
            # to be most accurate (near the training manifold).
            'enabled': True,
            # Sample: N(mean, (std_scale * std)^2) in raw/original units.
            'std_scale': 1.0,
            # Clip each dimension to mean +/- clip_n_std * std.
            'clip_n_std': 2.5,
            # Optional fixed seed for reproducibility.
            'seed': None,
        },
        'output': {
            #'result_filename': 'CVAE_lstm_300k_test.txt',
            'result_filename': 'train_dist_temp_transformer_300k_test.txt',
            # If None, defaults to result filename stem + '.pckl.gz'.
            'molecules_pickle_filename': None,
            # If None, defaults to result filename stem + '_quality_summary.csv'.
            'quality_summary_filename': None,
            #'sweep_stats_filename': 'CVAE_lstm_300k_test.csv',
            'sweep_stats_filename': 'train_dist_temp_transformer_300k_test.csv',
            
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
    # NOTE: In training-dist mode (run_training_dist=True), target properties vary per
    # sample, so we disable this fixed-target predicate.
    accept_predicate = None
    if bool(config.get('run_training_dist', False)):
        print('acceptance constraints: disabled for training-dist mode (targets vary per sample)')
    else:
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

    target_props_per_molecule = None
    target_prop_names = _default_prop_names(int(model_config['num_prop']))

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
        ms, smiles, pred_labels, target_props_per_molecule = converts_sweep_to_list_with_targets(
            sweep_results,
            prop_names=target_prop_names,
        )

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
    elif bool(config.get('run_training_dist', False)):
        # Sample target properties near training distribution (mean/std from training).
        if prop_norm_mean is None or prop_norm_std is None:
            raise ValueError('run_training_dist=True requires prop_norm_mean/std in model_config (saved during training).')

        seed = config.get('training_dist_seed')
        rng = np.random.default_rng(None if seed is None else int(seed))
        std_scale = float(config.get('training_dist_std_scale', 1.0))
        clip_n_std = config.get('training_dist_clip_n_std', 2.5)
        clip_n_std_val = None if clip_n_std is None else float(clip_n_std)

        run_scope = 'training_dist'
        if config['num_unique'] is not None:
            ms, smiles, run_stats, pred_labels, target_props_per_molecule = generate_unique_molecules_from_training_dist(
                model=model,
                charset=charset,
                start_codon=start_codon,
                seq_length=int(model_config['seq_length']),
                num_unique=int(config['num_unique']),
                max_batches=config['max_batches'],
                mean=float(model_config['mean']),
                stddev=float(model_config['stddev']),
                batch_size=int(model_config['batch_size']),
                latent_size=int(model_config['latent_size']),
                training_smiles=training_smiles,
                do_sample=bool(config['do_sample']),
                temperature=float(config['temperature']),
                top_k=top_k,
                strip_salts=bool(config.get('strip_salts', True)),
                decharge=bool(config.get('decharge', True)),
                canonicalize_tautomer=bool(config.get('canonicalize_tautomer', False)),
                prop_norm_mean=list(prop_norm_mean),
                prop_norm_std=list(prop_norm_std),
                std_scale=std_scale,
                clip_n_std=clip_n_std_val,
                rng=rng,
            )
        else:
            ms, smiles, run_stats, pred_labels, target_props_per_molecule = generate_fixed_iterations_from_training_dist(
                model=model,
                charset=charset,
                start_codon=start_codon,
                seq_length=int(model_config['seq_length']),
                num_iteration=int(config['num_iteration']),
                mean=float(model_config['mean']),
                stddev=float(model_config['stddev']),
                batch_size=int(model_config['batch_size']),
                latent_size=int(model_config['latent_size']),
                training_smiles=training_smiles,
                do_sample=bool(config['do_sample']),
                temperature=float(config['temperature']),
                top_k=top_k,
                strip_salts=bool(config.get('strip_salts', True)),
                decharge=bool(config.get('decharge', True)),
                canonicalize_tautomer=bool(config.get('canonicalize_tautomer', False)),
                prop_norm_mean=list(prop_norm_mean),
                prop_norm_std=list(prop_norm_std),
                std_scale=std_scale,
                clip_n_std=clip_n_std_val,
                rng=rng,
            )
    elif config['num_unique'] is not None:
        ms, smiles, run_stats, pred_labels = generate_unique_molecules(
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
        ms, smiles, run_stats, pred_labels = generate_fixed_iterations(
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
                'pred_labels': pred_labels,
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
    # NOTE: RDKit properties are still convenient for MW/LogP/TPSA, but the
    # optional 'pred_labels' columns let you attach arbitrary learned descriptors.
    if len(ms) == 0:
        df = pd.DataFrame({'smiles': [], 'MW': [], 'LogP': [], 'TPSA': []})
    else:
        desc = _compute_rdkit_descriptors(ms)
        df = pd.DataFrame({'smiles': smiles, 'MW': desc['MW'], 'LogP': desc['LogP'], 'TPSA': desc['TPSA']})

    # Attach the target properties that were used for conditioning.
    # - Single-target: the same target_row for all molecules.
    # - Sweep: keep per-molecule target row so downstream analysis is possible.
    if len(smiles) > 0:
        if target_props_per_molecule is None:
            target_props_per_molecule = [tuple(float(v) for v in target_row) for _ in range(len(smiles))]
        if len(target_props_per_molecule) == len(smiles):
            for j, pname in enumerate(target_prop_names):
                df[f'target_{pname}'] = [tp[j] for tp in target_props_per_molecule]

    if pred_labels is not None and len(smiles) > 0:
        if len(pred_labels) != len(smiles):
            print(
                'WARNING: pred_labels length does not match smiles length; '
                'skipping predicted label columns. '
                f'(pred_labels={len(pred_labels)} smiles={len(smiles)})'
            )
            pred_labels = None

    if pred_labels is not None and len(smiles) > 0:
        pred_arr = np.asarray(pred_labels, dtype=np.float32)
        if pred_arr.ndim == 1:
            pred_arr = pred_arr.reshape(-1, 1)
        for j in range(int(pred_arr.shape[1])):
            df[f'pred_label_{j}'] = pred_arr[:, j]

        # If the checkpoint contains label target metadata, try to expose human-friendly
        # `pred_<name>` columns.
        #
        # IMPORTANT: The label head can be trained either on:
        #   - normalized targets (same scale as conditioning `c`), or
        #   - raw/original property values.
        # We record this as `label_target_scale` in model_config.
        label_target_indices = model_config.get('label_target_indices')
        label_target_names = model_config.get('label_target_names')
        label_target_scale = str(model_config.get('label_target_scale', 'normalized')).lower()
        prop_norm_mean = model_config.get('prop_norm_mean')
        prop_norm_std = model_config.get('prop_norm_std')

        if label_target_indices is not None and len(label_target_indices) == int(pred_arr.shape[1]):
            idxs = [int(i) for i in label_target_indices]

            pred_out = pred_arr
            if label_target_scale == 'normalized':
                # Denormalize only when targets were normalized during training.
                if prop_norm_mean is None or prop_norm_std is None:
                    print('WARNING: label_target_scale=normalized but prop_norm_mean/std missing; skipping denorm.')
                    pred_out = pred_arr
                else:
                    mean_arr = np.asarray(prop_norm_mean, dtype=np.float32)[idxs]
                    std_arr = np.asarray(prop_norm_std, dtype=np.float32)[idxs]
                    std_arr = np.where(std_arr < 1e-8, 1.0, std_arr)
                    pred_out = (pred_arr * std_arr.reshape(1, -1)) + mean_arr.reshape(1, -1)

            for j in range(int(pred_out.shape[1])):
                if isinstance(label_target_names, list) and len(label_target_names) == int(pred_out.shape[1]):
                    col_name = str(label_target_names[j])
                else:
                    col_name = f'prop_{idxs[j]}'
                df[f'pred_{col_name}'] = pred_out[:, j]

            # Convenience comparisons when RDKit descriptor exists.
            if 'pred_LogP' in df.columns and 'LogP' in df.columns:
                df['pred_LogP_minus_rdkit_LogP'] = df['pred_LogP'] - df['LogP']
            if 'pred_MW' in df.columns and 'MW' in df.columns:
                df['pred_MW_minus_rdkit_MW'] = df['pred_MW'] - df['MW']

            # If we also have target columns, it can be useful to compare prediction vs target.
            if 'pred_LogP' in df.columns and 'target_LogP' in df.columns:
                df['pred_LogP_minus_target_LogP'] = df['pred_LogP'] - df['target_LogP']
            if 'pred_MW' in df.columns and 'target_MW' in df.columns:
                df['pred_MW_minus_target_MW'] = df['pred_MW'] - df['target_MW']

    print(df.describe())
    df.to_csv(config['result_filename'], index=False)

    end_time = t.time()
    print(f'time to run: {end_time - start}')

