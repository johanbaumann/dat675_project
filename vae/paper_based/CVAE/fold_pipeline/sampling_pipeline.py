from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, '..'))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from utils import (
    canonicalize_for_filtering,
    compose_train_config_from_dict,
    infer_training_config_path,
    load_condition_property_names,
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
    _collect_new_unique_from_raw_with_payloads,
    _compute_rdkit_descriptors,
    _new_stats,
    _print_quality_stats,
    _sample_batch_strings,
    _save_quality_summary_csv,
    create_and_restore_model,
    normalize_like_training,
    sample_target_props_like_training,
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


def _configure_rdkit_logging(*, suppress_parse_warnings: bool) -> None:
    if not bool(suppress_parse_warnings):
        return

    # RDKit logging API differs across versions/builds.
    # IMPORTANT: RDLogger.logger().setLevel() does NOT accept stdlib logging
    # levels (e.g. logging.CRITICAL == 50) in some RDKit versions; doing so can
    # trigger IndexError inside RDKit. Keep this best-effort and never fail
    # sampling because of logging configuration.

    disable_log = getattr(RDLogger, 'DisableLog', None)
    if callable(disable_log):
        # Most reliable way to silence RDKit across versions.
        try:
            disable_log('rdApp.*')
            return
        except Exception:
            pass
        # Fallback: disable common channels individually.
        try:
            for channel in ('rdApp.error', 'rdApp.warning', 'rdApp.info', 'rdApp.debug'):
                disable_log(channel)
            return
        except Exception:
            pass

    # Last-resort fallback for builds without DisableLog().
    # Use RDKit's own constants (if present), not stdlib logging constants.
    try:
        rdkit_level = getattr(RDLogger, 'CRITICAL', None)
        if rdkit_level is None:
            rdkit_level = getattr(RDLogger, 'ERROR', None)
        if rdkit_level is not None:
            RDLogger.logger().setLevel(rdkit_level)
    except Exception:
        pass


def _safe_murcko_scaffold_smiles(mol: Optional[Chem.Mol]) -> Optional[str]:
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None:
            return None
        return Chem.MolToSmiles(scaffold, isomericSmiles=False, canonical=True)
    except Exception:
        return None


def _load_blocked_scaffolds_from_csv(
    *,
    csv_path: str,
    smiles_column: str,
    strip_salts: bool,
    decharge: bool,
    canonicalize_tautomer: bool,
) -> set[str]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'test scaffold CSV does not exist: {csv_path}')

    df = pd.read_csv(csv_path)
    if smiles_column not in df.columns:
        raise ValueError(f"Missing smiles column '{smiles_column}' in test scaffold CSV: {csv_path}")

    blocked: set[str] = set()
    for raw in df[smiles_column].astype(str).tolist():
        can, mol, _ = canonicalize_for_filtering(
            raw,
            strip_salts=bool(strip_salts),
            decharge=bool(decharge),
            canonicalize_tautomer=bool(canonicalize_tautomer),
        )
        if can is None or mol is None:
            continue
        scaffold = _safe_murcko_scaffold_smiles(mol)
        if scaffold is not None and scaffold != '':
            blocked.add(scaffold)
    return blocked


def _select_generated_output_columns(df: pd.DataFrame, sampling_cfg: dict) -> pd.DataFrame:
    """Apply optional output-column selection for generated CSV export.

    Example:
      generated_outputs: ["smiles", "pred_pIC50"]
    """
    requested = sampling_cfg.get('generated_outputs', None)
    if requested is None:
        return df
    if not isinstance(requested, (list, tuple)):
        raise ValueError('sampling.generated_outputs must be a list of column names or null.')

    requested_cols = [str(c) for c in requested]
    selected_cols = [c for c in requested_cols if c in df.columns]
    missing_cols = [c for c in requested_cols if c not in df.columns]

    if len(missing_cols) > 0:
        raise ValueError(
            f'sampling.generated_outputs contains missing columns: {missing_cols}. '
            f'Available columns: {list(df.columns)}'
        )
    if len(selected_cols) == 0:
        raise ValueError('sampling.generated_outputs resolved to zero columns.')

    return df.loc[:, selected_cols].copy()


def run_sampling_for_fold(
    *,
    run_dir: str,
    output_dir: str,
    target_row: Optional[list[float]],
    sampling_cfg: dict,
    test_scaffold_csv: Optional[str] = None,
    test_scaffold_smiles_column: str = 'smiles',
) -> SamplingResult:
    os.makedirs(output_dir, exist_ok=True)
    _configure_rdkit_logging(
        suppress_parse_warnings=bool(sampling_cfg.get('suppress_rdkit_parse_errors', True))
    )

    checkpoint_glob = str(sampling_cfg.get('checkpoint_glob', 'model_best.ckpt-*.pt'))
    ckpt_path, model_config = _load_model_config_from_run(run_dir, checkpoint_glob=checkpoint_glob)

    charset, vocab, inferred_num_prop = load_sampling_metadata(
        model_config['prop_file'],
        int(model_config['seq_length']),
    )
    model_config['num_prop'] = int(inferred_num_prop)

    target_prop_names = model_config.get('condition_property_names')
    if not isinstance(target_prop_names, list) or len(target_prop_names) != int(model_config['num_prop']):
        target_prop_names = load_condition_property_names(
            model_config['prop_file'],
            int(model_config['num_prop']),
        )

    run_training_dist = bool(sampling_cfg.get('run_training_dist', False))
    if not run_training_dist:
        if target_row is None:
            raise ValueError('target_row is required when sampling.run_training_dist is false.')
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

    target_row_single: Optional[list[float]] = None
    if not run_training_dist:
        assert target_row is not None
        target_row_single = [float(v) for v in target_row]
        target_prop = _target_row_to_batch(target_row_single, int(model_config['batch_size']))
        target_prop = normalize_like_training(
            target_prop,
            model_config.get('prop_norm_mean'),
            model_config.get('prop_norm_std'),
        )
    else:
        target_prop = None

    prop_norm_mean = model_config.get('prop_norm_mean')
    prop_norm_std = model_config.get('prop_norm_std')
    dist_seed = sampling_cfg.get('training_dist_seed', None)
    dist_rng = np.random.default_rng(None if dist_seed is None else int(dist_seed))
    dist_std_scale = float(sampling_cfg.get('training_dist_std_scale', 1.0))
    dist_clip_raw = sampling_cfg.get('training_dist_clip_n_std', 2.5)
    dist_clip_n_std = None if dist_clip_raw is None else float(dist_clip_raw)
    if run_training_dist and (prop_norm_mean is None or prop_norm_std is None):
        raise ValueError(
            'sampling.run_training_dist=true requires model prop_norm_mean/std from training config/checkpoint.'
        )

    start_codon = np.array([np.array([vocab['X']]) for _ in range(int(model_config['batch_size']))])

    top_k_raw = sampling_cfg.get('top_k', 20)
    top_k = None if top_k_raw is None else int(top_k_raw)
    num_unique = int(sampling_cfg.get('num_unique', 1000))
    max_batches = int(sampling_cfg.get('max_batches', 5000))

    seen_smiles: set[str] = set()
    mol_by_smiles: dict[str, object] = {}
    pred_by_smiles: dict[str, np.ndarray] = {}
    target_by_smiles: dict[str, tuple[float, ...]] = {}
    total_stats = _new_stats()
    blocked_test_scaffolds: set[str] = set()
    rejected_by_test_scaffold = 0

    accept_predicate = None
    if not run_training_dist:
        assert target_row_single is not None
        accept_predicate = _build_accept_predicate(config=sampling_cfg, target_row=target_row_single)
    if bool(sampling_cfg.get('exclude_test_scaffolds', False)):
        source_csv = sampling_cfg.get('test_scaffold_csv') or test_scaffold_csv
        if not source_csv:
            raise ValueError(
                'exclude_test_scaffolds=True requires sampling.test_scaffold_csv or pipeline-provided test fold CSV.'
            )
        blocked_test_scaffolds = _load_blocked_scaffolds_from_csv(
            csv_path=str(source_csv),
            smiles_column=str(sampling_cfg.get('test_scaffold_smiles_column', test_scaffold_smiles_column)),
            strip_salts=bool(sampling_cfg.get('strip_salts', True)),
            decharge=bool(sampling_cfg.get('decharge', True)),
            canonicalize_tautomer=bool(sampling_cfg.get('canonicalize_tautomer', False)),
        )
        print(
            f'[sampling] test-scaffold exclusion enabled: '
            f'blocked_scaffolds={len(blocked_test_scaffolds)}, source={source_csv}'
        )
    else:
        print('[sampling] test-scaffold exclusion disabled')

    if run_training_dist:
        print(
            '[sampling] mode=training_dist '
            f'std_scale={dist_std_scale}, clip_n_std={dist_clip_n_std}, seed={dist_seed}'
        )
    else:
        print(f'[sampling] mode=single_target target_row={target_row_single}')

    for batch_idx in range(max_batches):
        batch_target_prop = target_prop
        batch_target_raw = None
        if run_training_dist:
            assert prop_norm_mean is not None and prop_norm_std is not None
            batch_target_raw = sample_target_props_like_training(
                batch_size=int(model_config['batch_size']),
                prop_norm_mean=list(prop_norm_mean),
                prop_norm_std=list(prop_norm_std),
                std_scale=dist_std_scale,
                clip_n_std=dist_clip_n_std,
                rng=dist_rng,
            )
            batch_target_prop = normalize_like_training(batch_target_raw, prop_norm_mean, prop_norm_std)

        assert batch_target_prop is not None
        raw_strings, _latent, pred_labels = _sample_batch_strings(
            model=model,
            charset=charset,
            target_prop=batch_target_prop,
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

        if run_training_dist:
            accepted, stats = _collect_new_unique_from_raw_with_payloads(
                raw_strings=raw_strings,
                payload_a=pred_labels,
                payload_b=batch_target_raw,
                seen_smiles=seen_smiles,
                training_smiles=training_smiles,
                eos_token='E',
                accept_predicate=None,
                require_neutral=bool(sampling_cfg.get('require_neutral', False)),
                strip_salts=bool(sampling_cfg.get('strip_salts', True)),
                decharge=bool(sampling_cfg.get('decharge', True)),
                canonicalize_tautomer=bool(sampling_cfg.get('canonicalize_tautomer', False)),
            )
        else:
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

        for accepted_row in accepted:
            if run_training_dist:
                can, mol, payload, payload_target_raw = accepted_row
            else:
                can, mol, payload = accepted_row
                payload_target_raw = None

            if blocked_test_scaffolds:
                scaffold = _safe_murcko_scaffold_smiles(mol)
                if scaffold is not None and scaffold in blocked_test_scaffolds:
                    total_stats['rejected_by_filter'] = int(total_stats.get('rejected_by_filter', 0)) + 1
                    rejected_by_test_scaffold += 1
                    continue
            if can not in mol_by_smiles:
                mol_by_smiles[can] = mol
                if payload is not None:
                    pred_by_smiles[can] = np.asarray(payload, dtype=np.float32)
                if payload_target_raw is not None:
                    target_by_smiles[can] = tuple(
                        float(v) for v in np.asarray(payload_target_raw, dtype=np.float32).reshape(-1).tolist()
                    )
            if len(mol_by_smiles) >= num_unique:
                break

        if batch_idx == 0 or (batch_idx + 1) % 10 == 0:
            print(
                f'[sampling] batches={batch_idx + 1}, unique_saved={len(mol_by_smiles)}/{num_unique}, '
                f"accepted={total_stats['accepted']}, invalid={total_stats['invalid_or_empty']}, "
                f"in_training={total_stats['in_training']}, duplicate={total_stats['duplicate']}, "
                f"rejected_test_scaffold={rejected_by_test_scaffold}"
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

    default_target_row = tuple(float('nan') for _ in range(int(model_config['num_prop'])))
    if run_training_dist:
        target_rows = [target_by_smiles.get(s, default_target_row) for s in smiles_out]
    else:
        assert target_row_single is not None
        row = tuple(float(v) for v in target_row_single)
        target_rows = [row for _ in smiles_out]

    for dim_idx, pname in enumerate(target_prop_names):
        out_df[f'target_{pname}'] = [r[dim_idx] for r in target_rows]

    if len(pred_by_smiles) > 0:
        pred_rows = [pred_by_smiles.get(s, None) for s in smiles_out]
        if all(p is not None for p in pred_rows):
            pred_rows_valid = [np.asarray(p, dtype=np.float32) for p in pred_rows if p is not None]
            pred_mat = np.stack(pred_rows_valid, axis=0)
            pred_names = model_config.get('label_target_names')
            if not isinstance(pred_names, list) or len(pred_names) != int(pred_mat.shape[1]):
                pred_names = sampling_cfg.get('pred_property_names', None)
            if not isinstance(pred_names, list) or len(pred_names) != int(pred_mat.shape[1]):
                pred_names = [f'pred_prop_{i}' for i in range(int(pred_mat.shape[1]))]
            for idx, name in enumerate(pred_names):
                out_df[f'pred_{name}'] = pred_mat[:, idx]

    # Optional user-controlled generated CSV schema.
    out_df = _select_generated_output_columns(out_df, sampling_cfg)

    generated_csv_path = os.path.abspath(
        os.path.join(output_dir, str(sampling_cfg.get('result_filename', 'generated.csv')))
    )
    save_generated_csv = bool(sampling_cfg.get('save_generated_csv', True))
    if save_generated_csv:
        out_df.to_csv(generated_csv_path, index=False)
    else:
        generated_csv_path = 'SKIPPED_GENERATED_CSV'
        print('[sampling] save_generated_csv=false, skipping generated CSV write')

    quality_summary_filename = str(sampling_cfg.get('quality_summary_filename', 'quality_summary.csv'))
    quality_cfg = dict(sampling_cfg)
    quality_cfg['result_filename'] = generated_csv_path
    quality_cfg['quality_summary_filename'] = os.path.abspath(os.path.join(output_dir, quality_summary_filename))
    save_quality_summary = bool(sampling_cfg.get('save_quality_summary', True))

    _print_quality_stats(total_stats, scope_label='Generated set quality (single fold)')
    if save_quality_summary:
        quality_summary_csv_path = _save_quality_summary_csv(
            stats=total_stats,
            run_scope=('training_dist' if run_training_dist else 'single_target'),
            num_molecules_saved=int(len(out_df)),
            config=quality_cfg,
        )
    else:
        quality_summary_csv_path = 'SKIPPED_QUALITY_SUMMARY'
        print('[sampling] save_quality_summary=false, skipping quality summary CSV write')
    if blocked_test_scaffolds:
        print(f'[sampling] rejected due to test-scaffold overlap: {rejected_by_test_scaffold}')

    debug_payload = {
        'run_dir': os.path.abspath(run_dir),
        'checkpoint_path': ckpt_path,
        'sampling_cfg': sampling_cfg,
        'run_training_dist': bool(run_training_dist),
        'target_row': (None if target_row_single is None else [float(v) for v in target_row_single]),
        'exclude_test_scaffolds': bool(sampling_cfg.get('exclude_test_scaffolds', False)),
        'num_blocked_test_scaffolds': int(len(blocked_test_scaffolds)),
        'rejected_by_test_scaffold': int(rejected_by_test_scaffold),
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
