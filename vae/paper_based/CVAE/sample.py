
from __future__ import annotations

import time as t
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.rdMolDescriptors import CalcTPSA

from model import CVAE
from utils import (
    collect_new_unique_from_raw,
    convert_to_smiles,
    load_data,
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
) -> list[str]:
    """Sample one batch and return decoded strings (still containing 'E' padding)."""
    latent_vector = np.random.normal(mean, stddev, (batch_size, latent_size))
    generated = model.sample(latent_vector, target_prop, start_codon, seq_length)
    return [convert_to_smiles(generated[i], charset) for i in range(len(generated))]


def _new_stats() -> dict:
    return {
        'total_generated': 0,
        'accepted': 0,
        'invalid_or_empty': 0,
        'in_training': 0,
        'duplicate': 0,
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
    not_ok = total_generated - accepted
    not_ok_share = (float(not_ok) / float(total_generated)) if total_generated > 0 else 0.0

    print(f'total generated molecules: {total_generated}')
    print(f'accepted molecules: {accepted}')
    print(f'not ok molecules: {not_ok} ({not_ok_share:.2%})')
    print(
        f'not ok breakdown -> invalid_or_empty: {invalid_or_empty}, '
        f'in_training: {in_training}, duplicate: {duplicate}'
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
        )
        total_trials += len(raw)
        batches += 1

        accepted, batch_stats = collect_new_unique_from_raw(
            raw_strings=raw,
            seen_smiles=seen_smiles,
            training_smiles=training_smiles,
            eos_token='E',
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
        )
        total_trials += len(raw)
        accepted, batch_stats = collect_new_unique_from_raw(
            raw_strings=raw,
            seen_smiles=seen_smiles,
            training_smiles=training_smiles,
            eos_token='E',
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


        start = t.time()

        # Single source of truth for run configuration.
        # Edit values here directly; CLI arguments are intentionally disabled.
        config = {
            'batch_size': 128,
            'num_iteration': 10,  # number of batches to sample (old behavior)
            'latent_size': 200,
            'unit_size': 512,
            'n_rnn_layer': 3,
            'seq_length': 120,
            'mean': 0.0,
            'stddev': 1.0,
            'num_prop': 3,
            'save_file': 'save/model_.ckpt-99.pt',
            'target_prop': '300.0 3.0 75.0',
            'prop_file': 'smiles_prop.txt',
            'result_filename': 'result.txt',
            'lr': 0.0001,
            'num_unique': 1000,
            'max_batches': 5000,
            # If True, molecules already present in training/property file are rejected.
            'exclude_training': True,
        }

        # Build vocabulary/charset from property file.
        _, _, charset, vocab, _, _ = load_data(config['prop_file'], config['seq_length'])
        vocab_size = len(charset)

        # Canonical training-set SMILES used for novelty filtering.
        if bool(config.get('exclude_training', True)):
            training_smiles = load_training_canonical_smiles(
                config['prop_file'],
                int(config['seq_length']),
            )
            print(f'training molecules available for exclusion: {len(training_smiles)}')
        else:
            training_smiles = set()
            print('training-set exclusion is disabled (exclude_training=False)')

        # Create and restore model.
        model = CVAE(vocab_size, config)
        model.restore(config['save_file'])
        print('Number of parameters : ', sum(p.numel() for p in model.parameters() if p.requires_grad))

        # Target property conditioning: replicate the target row for the whole batch.
        try:
            target_row = [float(p) for p in str(config['target_prop']).split()]
        except Exception:
            raise ValueError(
                'target_prop should be a string of space separated values. '
                'e.g. "300.0 3.0 75.0" for MW=300, LogP=3, TPSA=75'
            )

        target_prop = np.array([target_row for _ in range(int(config['batch_size']))], dtype=np.float32)

        # Start token: 'X'. In this dataset, 'X' is appended to the vocab in `load_data()`.
        start_codon = np.array([np.array([vocab['X']]) for _ in range(int(config['batch_size']))])

        if config['num_unique'] is not None:
            ms, smiles = generate_unique_molecules(
                model=model,
                charset=charset,
                target_prop=target_prop,
                start_codon=start_codon,
                seq_length=int(config['seq_length']),
                num_unique=int(config['num_unique']),
                max_batches=config['max_batches'],
                mean=float(config['mean']),
                stddev=float(config['stddev']),
                batch_size=int(config['batch_size']),
                latent_size=int(config['latent_size']),
                training_smiles=training_smiles,
            )
        else:
            ms, smiles = generate_fixed_iterations(
                model=model,
                charset=charset,
                target_prop=target_prop,
                start_codon=start_codon,
                seq_length=int(config['seq_length']),
                num_iteration=int(config['num_iteration']),
                mean=float(config['mean']),
                stddev=float(config['stddev']),
                batch_size=int(config['batch_size']),
                latent_size=int(config['latent_size']),
                training_smiles=training_smiles,
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

