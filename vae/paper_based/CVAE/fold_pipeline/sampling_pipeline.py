from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, '..'))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from utils import (
    compose_train_config_from_dict,
    infer_training_config_path,
    load_checkpoint_model_config,
    load_json,
    load_sampling_metadata,
    load_training_canonical_smiles,
    resolve_checkpoint_path,
)
from utils_labels import (
    _accumulate_stats,
    _build_accept_predicate,
    _collect_new_unique_from_raw_with_payload,
    _compute_rdkit_descriptors,
    _new_stats,
    _print_quality_stats,
    _sample_batch_strings,
    _save_quality_summary_csv,
    create_and_restore_model,
    normalize_like_training,
)


@dataclass(frozen=True)
class SamplingResult:
    run_dir: str
    checkpoint_path: str
    generated_csv_path: str
    quality_summary_csv_path: str
    num_saved: int
    stats: dict


def _load_model_config_from_run(run_dir: str, checkpoint_glob: str) -> tuple[str, dict]:
    ckpt = resolve_checkpoint_path(run_dir=run_dir, checkpoint_glob=checkpoint_glob)
    model_config = load_checkpoint_model_config(ckpt)
    if model_config is None:
        cfg_path = infer_training_config_path(ckpt)
        train_cfg = load_json(cfg_path)
        model_config = compose_train_config_from_dict(train_cfg)
    return str(ckpt), dict(model_config)


def _target_row_to_batch(target_row: list[float], batch_size: int) -> np.ndarray:
    return np.asarray([target_row for _ in range(int(batch_size))], dtype=np.float32)


def run_sampling_for_fold(
    *,
    run_dir: str,
    output_dir: str,
    target_row: list[float],
    sampling_cfg: dict,
) -> SamplingResult:
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_glob = str(sampling_cfg.get('checkpoint_glob', 'model_best.ckpt-*.pt'))
    ckpt_path, model_config = _load_model_config_from_run(run_dir, checkpoint_glob=checkpoint_glob)

    charset, vocab, inferred_num_prop = load_sampling_metadata(
        model_config['prop_file'],
        int(model_config['seq_length']),
    )
    model_config['num_prop'] = int(inferred_num_prop)

    if len(target_row) != int(model_config['num_prop']):
        raise ValueError(
            f'target_row has {len(target_row)} values, model expects {int(model_config["num_prop"])} properties.'
        )

    if bool(sampling_cfg.get('exclude_training', True)):
        training_smiles = load_training_canonical_smiles(
            model_config['prop_file'],
            int(model_config['seq_length']),
            strip_salts=bool(sampling_cfg.get('strip_salts', True)),
            decharge=bool(sampling_cfg.get('decharge', True)),
            canonicalize_tautomer=bool(sampling_cfg.get('canonicalize_tautomer', False)),
        )
    else:
        training_smiles = set()

    model = create_and_restore_model({'save_file': ckpt_path}, model_config, len(charset))

    target_row = [float(v) for v in target_row]
    target_prop = _target_row_to_batch(target_row, int(model_config['batch_size']))
    target_prop = normalize_like_training(
        target_prop,
        model_config.get('prop_norm_mean'),
        model_config.get('prop_norm_std'),
    )

    start_codon = np.array([np.array([vocab['X']]) for _ in range(int(model_config['batch_size']))])

    top_k_raw = sampling_cfg.get('top_k', 20)
    top_k = None if top_k_raw is None else int(top_k_raw)
    num_unique = int(sampling_cfg.get('num_unique', 1000))
    max_batches = int(sampling_cfg.get('max_batches', 5000))

    seen_smiles: set[str] = set()
    mol_by_smiles: dict[str, object] = {}
    pred_by_smiles: dict[str, np.ndarray] = {}
    total_stats = _new_stats()

    accept_predicate = _build_accept_predicate(config=sampling_cfg, target_row=target_row)

    for batch_idx in range(max_batches):
        raw_strings, _latent, pred_labels = _sample_batch_strings(
            model=model,
            charset=charset,
            target_prop=target_prop,
            start_codon=start_codon,
            seq_length=int(model_config['seq_length']),
            mean=float(model_config['mean']),
            stddev=float(model_config['stddev']),
            batch_size=int(model_config['batch_size']),
            latent_size=int(model_config['latent_size']),
            do_sample=bool(sampling_cfg.get('do_sample', True)),
            temperature=float(sampling_cfg.get('temperature', 0.7)),
            top_k=top_k,
        )

        accepted, stats = _collect_new_unique_from_raw_with_payload(
            raw_strings=raw_strings,
            payload=pred_labels,
            seen_smiles=seen_smiles,
            training_smiles=training_smiles,
            eos_token='E',
            accept_predicate=accept_predicate,
            require_neutral=bool(sampling_cfg.get('require_neutral', False)),
            strip_salts=bool(sampling_cfg.get('strip_salts', True)),
            decharge=bool(sampling_cfg.get('decharge', True)),
            canonicalize_tautomer=bool(sampling_cfg.get('canonicalize_tautomer', False)),
        )
        _accumulate_stats(total_stats, stats)

        for can, mol, payload in accepted:
            if can not in mol_by_smiles:
                mol_by_smiles[can] = mol
                if payload is not None:
                    pred_by_smiles[can] = np.asarray(payload, dtype=np.float32)
            if len(mol_by_smiles) >= num_unique:
                break

        if batch_idx == 0 or (batch_idx + 1) % 10 == 0:
            print(
                f'[sampling] batches={batch_idx + 1}, unique_saved={len(mol_by_smiles)}/{num_unique}, '
                f"accepted={total_stats['accepted']}, invalid={total_stats['invalid_or_empty']}, "
                f"in_training={total_stats['in_training']}, duplicate={total_stats['duplicate']}"
            )

        if len(mol_by_smiles) >= num_unique:
            break

    smiles_out = sorted(mol_by_smiles.keys())[:num_unique]
    mols_out = [mol_by_smiles[s] for s in smiles_out]
    desc = _compute_rdkit_descriptors(mols_out)

    out_df = pd.DataFrame(
        {
            'smiles': smiles_out,
            'MW': desc.get('MW', []),
            'LogP': desc.get('LogP', []),
            'TPSA': desc.get('TPSA', []),
        }
    )

    for dim_idx, val in enumerate(target_row):
        out_df[f'target_prop_{dim_idx}'] = float(val)

    if len(pred_by_smiles) > 0:
        pred_rows = [pred_by_smiles.get(s, None) for s in smiles_out]
        if all(p is not None for p in pred_rows):
            pred_rows_valid = [np.asarray(p, dtype=np.float32) for p in pred_rows if p is not None]
            pred_mat = np.stack(pred_rows_valid, axis=0)
            pred_names = model_config.get('label_target_names')
            if not isinstance(pred_names, list) or len(pred_names) != int(pred_mat.shape[1]):
                pred_names = [f'pred_prop_{i}' for i in range(int(pred_mat.shape[1]))]
            for idx, name in enumerate(pred_names):
                out_df[f'pred_{name}'] = pred_mat[:, idx]

    generated_csv_path = os.path.abspath(
        os.path.join(output_dir, str(sampling_cfg.get('result_filename', 'generated.csv')))
    )
    out_df.to_csv(generated_csv_path, index=False)

    quality_summary_filename = str(sampling_cfg.get('quality_summary_filename', 'quality_summary.csv'))
    quality_cfg = dict(sampling_cfg)
    quality_cfg['result_filename'] = generated_csv_path
    quality_cfg['quality_summary_filename'] = os.path.abspath(os.path.join(output_dir, quality_summary_filename))

    _print_quality_stats(total_stats, scope_label='Generated set quality (single fold)')
    quality_summary_csv_path = _save_quality_summary_csv(
        stats=total_stats,
        run_scope='single_target',
        num_molecules_saved=int(len(out_df)),
        config=quality_cfg,
    )

    debug_payload = {
        'run_dir': os.path.abspath(run_dir),
        'checkpoint_path': ckpt_path,
        'sampling_cfg': sampling_cfg,
        'target_row': [float(v) for v in target_row],
        'num_saved': int(len(out_df)),
        'stats': total_stats,
        'generated_csv_path': generated_csv_path,
        'quality_summary_csv_path': quality_summary_csv_path,
    }
    with open(os.path.join(output_dir, 'sampling_debug.json'), 'w', encoding='utf-8') as f:
        json.dump(debug_payload, f, indent=2)

    return SamplingResult(
        run_dir=os.path.abspath(run_dir),
        checkpoint_path=str(ckpt_path),
        generated_csv_path=str(generated_csv_path),
        quality_summary_csv_path=str(quality_summary_csv_path),
        num_saved=int(len(out_df)),
        stats=dict(total_stats),
    )
