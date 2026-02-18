
from __future__ import annotations

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
    load_data,
    load_checkpoint_model_config,
    load_json,
    load_training_canonical_smiles,
)


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
    }


def _accumulate_stats(total: dict, inc: dict) -> None:
    for key in total.keys():
        total[key] += int(inc.get(key, 0))


def _print_quality_stats(stats: dict) -> None:
    total_generated = int(stats['total_generated'])
    accepted = int(stats['accepted'])
    invalid_or_empty = int(stats['invalid_or_empty'])
    in_training = int(stats['in_training'])
    duplicate = int(stats['duplicate'])
    rejected_by_filter = int(stats.get('rejected_by_filter', 0))
    not_ok = total_generated - accepted
    not_ok_share = (float(not_ok) / float(total_generated)) if total_generated > 0 else 0.0

    print(f'total generated molecules: {total_generated}')
    print(f'accepted molecules: {accepted}')
    print(f'not ok molecules: {not_ok} ({not_ok_share:.2%})')
    print(
        f'not ok breakdown -> invalid_or_empty: {invalid_or_empty}, '
        f'in_training: {in_training}, duplicate: {duplicate}'
    )
    if rejected_by_filter:
        print(f'additional rejected_by_filter: {rejected_by_filter}')


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
    accept_predicate: Optional[Callable[[str, Chem.Mol], bool]] = None,
) -> tuple[list[Chem.Mol], list[str]]:
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
                f"in_training: {batch_stats['in_training']}, duplicate: {batch_stats['duplicate']}, rejected_by_filter: {batch_stats.get('rejected_by_filter', 0)}"
            )

    _print_quality_stats(total_stats)

    smiles_out = sorted(unique_mols_by_smiles.keys())[:num_unique]
    mols_out = [unique_mols_by_smiles[s] for s in smiles_out]
    return mols_out, smiles_out


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
    accept_predicate: Optional[Callable[[str, Chem.Mol], bool]] = None,
) -> tuple[list[Chem.Mol], list[str]]:
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
        )
        _accumulate_stats(total_stats, batch_stats)

        for can, mol in accepted:
            unique_mols_by_smiles.setdefault(can, mol)

    print(f'number of generated trials : {total_trials}')
    print(f'number of unique valid molecules : {len(unique_mols_by_smiles)}')
    _print_quality_stats(total_stats)
    smiles_out = sorted(unique_mols_by_smiles.keys())
    mols_out = [unique_mols_by_smiles[s] for s in smiles_out]
    return mols_out, smiles_out



if __name__ == '__main__':

    # Silence RDKit parse error spam (we still count invalid SMILES).
    RDLogger.DisableLog('rdApp.error')
    RDLogger.DisableLog('rdApp.warning')

    start = t.time()

    # Runtime sampling options.
    # Model architecture/training hyperparameters are loaded from training_config.json.
    config = {
        #'batch_size': 64, # for Transformer,
        'batch_size': 128, # for C-VAE (paper used 256, but that may cause OOM on smaller GPUs)
        'num_iteration': 10,  # number of batches to sample (old behavior)
        'save_file': 'save/model_best.ckpt-37.pt', #for C-VAE
        #"save_file": 'save/best_model_trans.pt', # for transformer!
        #'training_config_file': 'trans_config', # for transformer!
        'training_config_file': None,
        'target_prop': '300.0 3.0',
        'prop_file': None,
        'seq_length': None,
        'mean': None,
        'stddev': None,
        'result_filename': 'CVAE_result.txt',
        'num_unique': 1000,
        'max_batches': 5000,
        # Sampling controls. Greedy decoding (do_sample=False) often collapses to 1 molecule.
        'do_sample': False,
        # Sweep for this checkpoint suggests ~temperature=0.6, top_k=20 gives much higher unique+novel acceptance.
        'temperature': 0.6, # higher temperature -> more random, lower temperature -> more valid, less diverse
        'top_k': 20, # limits sampling to the top_k most probable tokens at each step. Can help improve validity at low temperatures.

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
    }

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

    # Build vocabulary/charset from property file.
    _, _, charset, vocab, loaded_labels, _ = load_data(model_config['prop_file'], int(model_config['seq_length']))
    vocab_size = len(charset)
    inferred_num_prop = int(loaded_labels.shape[1])
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
        )
        print(f'training molecules available for exclusion: {len(training_smiles)}')
    else:
        training_smiles = set()
        print('training-set exclusion is disabled (exclude_training=False)')

    # Create and restore model.
    model = CVAE(vocab_size, model_config)
    model.restore(config['save_file'])
    print('Number of parameters : ', sum(p.numel() for p in model.parameters() if p.requires_grad))

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

    # If training used standardized properties, apply the same transform here.
    prop_norm_mean = model_config.get('prop_norm_mean')
    prop_norm_std = model_config.get('prop_norm_std')
    if prop_norm_mean is not None and prop_norm_std is not None:
        mean_arr = np.array(prop_norm_mean, dtype=np.float32)
        std_arr = np.array(prop_norm_std, dtype=np.float32)
        std_arr = np.where(std_arr < 1e-8, 1.0, std_arr)
        target_prop = (target_prop - mean_arr) / std_arr

    # Start token: 'X'. In this dataset, 'X' is appended to the vocab in `load_data()`.
    start_codon = np.array([np.array([vocab['X']]) for _ in range(int(model_config['batch_size']))])

    top_k_val = config.get('top_k')
    top_k = (None if top_k_val is None else int(top_k_val))

    if config['num_unique'] is not None:
        ms, smiles = generate_unique_molecules(
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
            accept_predicate=accept_predicate,
        )
    else:
        ms, smiles = generate_fixed_iterations(
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
            accept_predicate=accept_predicate,
        )

    print('number of valid smiles : ', len(ms))

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

