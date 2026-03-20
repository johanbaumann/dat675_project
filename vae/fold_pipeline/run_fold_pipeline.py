from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from typing import Optional

import numpy as np

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, '..'))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from fold_pipeline.fold_data import convert_cv_iteration_to_prop_files, discover_cv_fold_iterations
from fold_pipeline.sampling_pipeline import SamplingResult, run_sampling_for_iteration
from analysis_modules import load_analysis_config_from_file, run_analysis_pipeline


DEFAULT_CONFIG_PATH = os.path.join(_THIS_DIR, 'fold_pipeline_config.example.json')


def _deep_update_dict(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update_dict(out[key], value)
        else:
            out[key] = value
    return out


def _read_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f'Config must be a JSON object: {path}')
    return payload


def _resolve_optional_path(path_value: Optional[str], *, base_dir: str) -> Optional[str]:
    if path_value is None:
        return None
    raw = str(path_value).strip()
    if raw == '':
        return None
    if os.path.isabs(raw):
        return os.path.abspath(raw)
    return os.path.abspath(os.path.join(base_dir, raw))


def _resolve_required_path(path_value: Optional[str], *, base_dir: str, key_name: str) -> str:
    resolved = _resolve_optional_path(path_value, base_dir=base_dir)
    if resolved is None:
        raise KeyError(f'Config key is required and cannot be empty: {key_name}')
    return str(resolved)


def _resolve_python_executable(value: Optional[str], *, workspace_root: str) -> str:
    if value is None:
        return str(sys.executable)
    raw = str(value).strip()
    if raw == '':
        return str(sys.executable)

    # Keep command-name executables (e.g., "python", "python3") intact;
    # resolve path-like values relative to workspace_root.
    if os.path.sep in raw or '/' in raw:
        return _resolve_required_path(raw, base_dir=workspace_root, key_name='python_executable')
    return raw


def _resolve_script_path(value: Optional[str], *, workspace_root: str, key_name: str) -> str:
    script_path = _resolve_required_path(value, base_dir=workspace_root, key_name=key_name)
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f'Configured script for {key_name} was not found: {script_path}')
    return script_path


def _write_json(path: str, payload: dict) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    return path


def _safe_remove_path(path: str) -> None:
    if not path:
        return
    if not os.path.exists(path):
        return
    if os.path.isdir(path):
        shutil.rmtree(path)
        return
    os.remove(path)


def _resolve_latest_checkpoint_in_run_dir(*, run_dir: str, checkpoint_glob: str) -> Optional[str]:
    if not os.path.isdir(run_dir):
        return None

    primary_pattern = os.path.join(run_dir, str(checkpoint_glob))
    primary_matches = [p for p in glob.glob(primary_pattern) if os.path.isfile(p)]
    if primary_matches:
        primary_matches.sort(key=os.path.getmtime, reverse=True)
        return os.path.abspath(primary_matches[0])

    fallback_pattern = os.path.join(run_dir, '*.pt')
    fallback_matches = [p for p in glob.glob(fallback_pattern) if os.path.isfile(p)]
    if fallback_matches:
        fallback_matches.sort(key=os.path.getmtime, reverse=True)
        return os.path.abspath(fallback_matches[0])

    return None


def _verify_training_stage_outputs(
    *,
    fold_name: str,
    train_run_dir: str,
    train_cfg_path: str,
    checkpoint_glob: str,
    analysis_enabled: bool,
) -> tuple[str, Optional[str]]:
    """Verify required artifacts after training stage completes.

    Returns:
      (checkpoint_path, prediction_eval_csv_path_or_none)
    """
    if not os.path.isdir(train_run_dir):
        raise RuntimeError(
            f'{fold_name}: training run directory does not exist after train stage: {train_run_dir}'
        )

    training_config_json = os.path.join(train_run_dir, 'training_config.json')
    if not os.path.isfile(training_config_json):
        raise RuntimeError(
            f'{fold_name}: training completed but training_config.json is missing at {training_config_json}. '
            'This usually indicates training failed before model initialization was persisted.'
        )

    checkpoint_path = _resolve_latest_checkpoint_in_run_dir(
        run_dir=train_run_dir,
        checkpoint_glob=checkpoint_glob,
    )
    if checkpoint_path is None:
        raise RuntimeError(
            f'{fold_name}: training stage exited but no checkpoint was written under {train_run_dir} '
            f'(pattern={checkpoint_glob!r}). '
            'Possible causes: early numerical failure/non-finite loss or checkpoint saving disabled.'
        )

    pred_eval_path = _resolve_prediction_eval_csv_path(
        train_dir=train_run_dir,
        train_cfg_path=train_cfg_path,
    )
    has_pred_eval = bool(pred_eval_path and os.path.isfile(pred_eval_path))
    if analysis_enabled:
        print(
            f'[{fold_name}] training output check: checkpoint={checkpoint_path}, '
            f'prediction_eval_csv_present={has_pred_eval}'
        )
    else:
        print(f'[{fold_name}] training output check: checkpoint={checkpoint_path}')

    return str(checkpoint_path), (str(pred_eval_path) if has_pred_eval else None)


def _purge_python_caches(root_dir: str) -> int:
    removed = 0
    for dirpath, dirnames, _filenames in os.walk(root_dir):
        for dirname in list(dirnames):
            if dirname != '__pycache__':
                continue
            cache_path = os.path.join(dirpath, dirname)
            try:
                shutil.rmtree(cache_path)
                removed += 1
            except Exception:
                pass
    return int(removed)


def _read_first_csv_row(path: str) -> dict:
    with open(path, 'r', encoding='utf-8', errors='ignore', newline='') as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
    if row is None:
        raise ValueError(f'CSV has no data rows: {path}')
    return dict(row)


def _count_csv_data_rows(path: str) -> int:
    with open(path, 'r', encoding='utf-8', errors='ignore', newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return 0
        return int(sum(1 for _ in reader))


def _to_int(value, *, default: int = 0) -> int:
    if value is None:
        return int(default)
    raw = str(value).strip()
    if raw == '':
        return int(default)
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _to_float(value, *, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    raw = str(value).strip()
    if raw == '':
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _compute_vun_from_counts(
    *,
    total_generated: int,
    invalid_or_empty: int,
    discarded_cleanup: int,
    in_training: int,
    duplicate: int,
    accepted: int,
) -> dict:
    total = int(total_generated)
    if total <= 0:
        return {
            'validity': 0.0,
            'uniqueness': 0.0,
            'novelty': 0.0,
            'acceptance_rate': 0.0,
            'valid_count': 0,
            'unique_count': 0,
            'novel_count': 0,
        }

    valid_count = int(total - int(invalid_or_empty) - int(discarded_cleanup))
    unique_count = int(total - int(duplicate))
    novel_count = int(total - int(in_training))
    accepted_count = int(accepted)

    return {
        'validity': float(valid_count) / float(total),
        'uniqueness': float(unique_count) / float(total),
        'novelty': float(novel_count) / float(total),
        'acceptance_rate': float(accepted_count) / float(total),
        'valid_count': int(valid_count),
        'unique_count': int(unique_count),
        'novel_count': int(novel_count),
    }


def _resolve_existing_sampling_result_for_analysis_only(
    *,
    fold_name: str,
    train_run_dir: str,
    generated_csv_path: str,
    quality_summary_csv_path: str,
) -> SamplingResult:
    if not os.path.isfile(generated_csv_path):
        raise RuntimeError(
            f'{fold_name}: sampling is disabled but no generated CSV was found at: {generated_csv_path}. '
            'Set sampling.enabled=true to generate molecules first, or point artifacts_output_root to existing outputs.'
        )

    num_saved = _count_csv_data_rows(generated_csv_path)
    if int(num_saved) <= 0:
        raise RuntimeError(
            f'{fold_name}: sampling is disabled and generated CSV has zero molecules: {generated_csv_path}. '
            'Hard-failing to avoid running analysis on empty generated data.'
        )

    if not os.path.isfile(quality_summary_csv_path):
        print(
            f'[{fold_name}] NOTE: expected quality summary is missing: {quality_summary_csv_path}. '
            'Cross-fold V.U.N aggregation will skip this fold unless the file exists.'
        )

    return SamplingResult(
        run_dir=os.path.abspath(train_run_dir),
        checkpoint_path='REUSED_EXISTING_GENERATED_CSV',
        generated_csv_path=os.path.abspath(generated_csv_path),
        quality_summary_csv_path=os.path.abspath(quality_summary_csv_path),
        num_saved=int(num_saved),
        stats={},
    )


def _write_cross_fold_analysis_summary(
    *,
    artifacts_root: str,
    global_manifest: dict,
) -> Optional[str]:
    iterations = list(global_manifest.get('iterations', []))
    if len(iterations) == 0:
        return None

    total_generated = 0
    accepted = 0
    invalid_or_empty = 0
    discarded_cleanup = 0
    in_training = 0
    duplicate = 0
    rejected_by_filter = 0
    salt_stripped = 0
    tautomer_canonicalized = 0

    diversity_weighted_sum = 0.0
    diversity_weight_sum = 0
    mean_tanimoto_weighted_sum = 0.0
    mean_tanimoto_weight_sum = 0
    internal_diversity_weighted_sum = 0.0
    internal_diversity_weight_sum = 0
    internal_mean_tanimoto_weighted_sum = 0.0
    internal_mean_tanimoto_weight_sum = 0

    per_fold: list[dict] = []
    folds_with_quality = 0
    folds_with_diversity = 0
    folds_with_internal_diversity = 0

    for fold in iterations:
        fold_name = str(fold.get('iteration_name', 'unknown_fold'))
        sampling_result = dict(fold.get('sampling_result', {}) or {})
        quality_path = str(sampling_result.get('quality_summary_csv_path', '')).strip()
        generated_rows = int(sampling_result.get('num_saved', 0) or 0)

        fold_row = {
            'iteration_name': fold_name,
            'quality_summary_csv_path': quality_path,
            'generated_rows': int(generated_rows),
            'validity': None,
            'uniqueness': None,
            'novelty': None,
            'acceptance_rate': None,
            'diversity_score': None,
            'mean_tanimoto_all_pairs': None,
            'diversity_internal_score': None,
            'mean_tanimoto_internal_all_pairs': None,
            'internal_similarity_mode': None,
            'internal_similarity_pairs_used': None,
            'internal_similarity_pairs_total': None,
            'pred_vs_ref_mae': None,
            'pred_vs_ref_std_ae': None,
            'pred_vs_ref_count': None,
            'target_vs_ref_mae': None,
            'target_vs_ref_std_ae': None,
            'target_vs_ref_count': None,
        }

        if quality_path and os.path.isfile(quality_path):
            row = _read_first_csv_row(quality_path)
            row_total_generated = _to_int(row.get('total_generated'))
            row_accepted = _to_int(row.get('accepted'))
            row_invalid = _to_int(row.get('invalid_or_empty'))
            row_discarded = _to_int(row.get('discarded_cleanup'))
            row_in_training = _to_int(row.get('in_training'))
            row_duplicate = _to_int(row.get('duplicate'))
            row_rejected = _to_int(row.get('rejected_by_filter'))
            row_salt = _to_int(row.get('salt_stripped'))
            row_tautomer = _to_int(row.get('tautomer_canonicalized'))

            total_generated += int(row_total_generated)
            accepted += int(row_accepted)
            invalid_or_empty += int(row_invalid)
            discarded_cleanup += int(row_discarded)
            in_training += int(row_in_training)
            duplicate += int(row_duplicate)
            rejected_by_filter += int(row_rejected)
            salt_stripped += int(row_salt)
            tautomer_canonicalized += int(row_tautomer)
            folds_with_quality += 1

            vun_fold = _compute_vun_from_counts(
                total_generated=row_total_generated,
                invalid_or_empty=row_invalid,
                discarded_cleanup=row_discarded,
                in_training=row_in_training,
                duplicate=row_duplicate,
                accepted=row_accepted,
            )
            fold_row['validity'] = float(vun_fold['validity'])
            fold_row['uniqueness'] = float(vun_fold['uniqueness'])
            fold_row['novelty'] = float(vun_fold['novelty'])
            fold_row['acceptance_rate'] = float(vun_fold['acceptance_rate'])

        analysis_summary_path = str(fold.get('analysis_summary_path', '')).strip()
        if analysis_summary_path and os.path.isfile(analysis_summary_path):
            payload = _read_json(analysis_summary_path)
            summary = dict(payload.get('summary', {}) or {})
            div = _to_float(summary.get('diversity_score'))
            mean_sim = _to_float(summary.get('mean_tanimoto_all_pairs'))
            div_internal = _to_float(summary.get('diversity_internal_score'))
            mean_sim_internal = _to_float(summary.get('mean_tanimoto_internal_all_pairs'))
            internal_mode = summary.get('internal_similarity_mode')
            internal_pairs_used = _to_int(summary.get('internal_similarity_pairs_used'), default=0)
            internal_pairs_total = _to_int(summary.get('internal_similarity_pairs_total'), default=0)
            n_rows = _to_int(summary.get('num_generated_rows'), default=generated_rows)
            pred_mae = _to_float(summary.get('pred_vs_ref_mae'))
            pred_std_ae = _to_float(summary.get('pred_vs_ref_std_ae'))
            pred_count = _to_int(summary.get('pred_vs_ref_count'), default=0)
            target_mae = _to_float(summary.get('target_vs_ref_mae'))
            target_std_ae = _to_float(summary.get('target_vs_ref_std_ae'))
            target_count = _to_int(summary.get('target_vs_ref_count'), default=0)
            if div is not None and n_rows > 0:
                diversity_weighted_sum += float(div) * float(n_rows)
                diversity_weight_sum += int(n_rows)
                fold_row['diversity_score'] = float(div)
                folds_with_diversity += 1
            if mean_sim is not None and n_rows > 0:
                mean_tanimoto_weighted_sum += float(mean_sim) * float(n_rows)
                mean_tanimoto_weight_sum += int(n_rows)
                fold_row['mean_tanimoto_all_pairs'] = float(mean_sim)
            if div_internal is not None and n_rows > 0:
                internal_diversity_weighted_sum += float(div_internal) * float(n_rows)
                internal_diversity_weight_sum += int(n_rows)
                fold_row['diversity_internal_score'] = float(div_internal)
                folds_with_internal_diversity += 1
            if mean_sim_internal is not None and n_rows > 0:
                internal_mean_tanimoto_weighted_sum += float(mean_sim_internal) * float(n_rows)
                internal_mean_tanimoto_weight_sum += int(n_rows)
                fold_row['mean_tanimoto_internal_all_pairs'] = float(mean_sim_internal)
            if internal_mode is not None:
                fold_row['internal_similarity_mode'] = str(internal_mode)
            if internal_pairs_used > 0:
                fold_row['internal_similarity_pairs_used'] = int(internal_pairs_used)
            if internal_pairs_total > 0:
                fold_row['internal_similarity_pairs_total'] = int(internal_pairs_total)
            if pred_mae is not None:
                fold_row['pred_vs_ref_mae'] = float(pred_mae)
            if pred_std_ae is not None:
                fold_row['pred_vs_ref_std_ae'] = float(pred_std_ae)
            if pred_count > 0:
                fold_row['pred_vs_ref_count'] = int(pred_count)
            if target_mae is not None:
                fold_row['target_vs_ref_mae'] = float(target_mae)
            if target_std_ae is not None:
                fold_row['target_vs_ref_std_ae'] = float(target_std_ae)
            if target_count > 0:
                fold_row['target_vs_ref_count'] = int(target_count)

        per_fold.append(fold_row)

    vun_agg = _compute_vun_from_counts(
        total_generated=total_generated,
        invalid_or_empty=invalid_or_empty,
        discarded_cleanup=discarded_cleanup,
        in_training=in_training,
        duplicate=duplicate,
        accepted=accepted,
    )

    mean_tanimoto_weighted = (
        float(mean_tanimoto_weighted_sum) / float(mean_tanimoto_weight_sum)
        if mean_tanimoto_weight_sum > 0
        else None
    )
    diversity_weighted = (
        float(diversity_weighted_sum) / float(diversity_weight_sum)
        if diversity_weight_sum > 0
        else None
    )
    internal_mean_tanimoto_weighted = (
        float(internal_mean_tanimoto_weighted_sum) / float(internal_mean_tanimoto_weight_sum)
        if internal_mean_tanimoto_weight_sum > 0
        else None
    )
    internal_diversity_weighted = (
        float(internal_diversity_weighted_sum) / float(internal_diversity_weight_sum)
        if internal_diversity_weight_sum > 0
        else None
    )

    payload = {
        'num_iterations': int(len(iterations)),
        'num_folds_with_quality_summary': int(folds_with_quality),
        'num_folds_with_diversity_summary': int(folds_with_diversity),
        'num_folds_with_internal_diversity_summary': int(folds_with_internal_diversity),
        'quality_counts': {
            'total_generated': int(total_generated),
            'accepted': int(accepted),
            'invalid_or_empty': int(invalid_or_empty),
            'discarded_cleanup': int(discarded_cleanup),
            'in_training': int(in_training),
            'duplicate': int(duplicate),
            'rejected_by_filter': int(rejected_by_filter),
            'salt_stripped': int(salt_stripped),
            'tautomer_canonicalized': int(tautomer_canonicalized),
        },
        'vun_aggregated': {
            'validity': float(vun_agg['validity']),
            'uniqueness': float(vun_agg['uniqueness']),
            'novelty': float(vun_agg['novelty']),
            'acceptance_rate': float(vun_agg['acceptance_rate']),
            'valid_count': int(vun_agg['valid_count']),
            'unique_count': int(vun_agg['unique_count']),
            'novel_count': int(vun_agg['novel_count']),
        },
        'diversity_aggregated': {
            # Diversity follows existing analysis definition: diversity = 1 - mean_tanimoto.
            'weighted_mean_tanimoto_all_pairs': mean_tanimoto_weighted,
            'weighted_diversity_score': diversity_weighted,
            'weight_total_generated_rows': int(diversity_weight_sum),
            'weighted_mean_tanimoto_external_all_pairs': mean_tanimoto_weighted,
            'weighted_diversity_external_score': diversity_weighted,
            'weight_total_generated_rows_external': int(diversity_weight_sum),
            'weighted_mean_tanimoto_internal_all_pairs': internal_mean_tanimoto_weighted,
            'weighted_diversity_internal_score': internal_diversity_weighted,
            'weight_total_generated_rows_internal': int(internal_diversity_weight_sum),
        },
        'per_fold': per_fold,
    }

    out_path = _write_json(os.path.join(artifacts_root, 'cross_fold_analysis_summary.json'), payload)
    print(f'[analysis] cross-fold summary written: {out_path}')
    return out_path


def _write_cv_combo_error_summary_csv(*, cross_fold_summary_path: str, artifacts_root: str) -> Optional[str]:
    payload = _read_json(cross_fold_summary_path)
    per_fold = list(payload.get('per_fold', []) or [])
    if len(per_fold) == 0:
        return None

    out_dir = os.path.join(artifacts_root, 'cv_combo')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.abspath(os.path.join(out_dir, 'cv_combo_error_stats.csv'))

    fieldnames = [
        'iteration_name',
        'pred_vs_ref_count',
        'pred_vs_ref_mae',
        'pred_vs_ref_std_ae',
        'target_vs_ref_count',
        'target_vs_ref_mae',
        'target_vs_ref_std_ae',
        'mae_gap_target_minus_pred',
    ]

    pred_weighted_sum = 0.0
    pred_weight_total = 0
    target_weighted_sum = 0.0
    target_weight_total = 0
    rows: list[dict] = []

    for row in per_fold:
        pred_count = _to_int(row.get('pred_vs_ref_count'), default=0)
        pred_mae = _to_float(row.get('pred_vs_ref_mae'))
        pred_std = _to_float(row.get('pred_vs_ref_std_ae'))
        target_count = _to_int(row.get('target_vs_ref_count'), default=0)
        target_mae = _to_float(row.get('target_vs_ref_mae'))
        target_std = _to_float(row.get('target_vs_ref_std_ae'))

        if pred_mae is not None and pred_count > 0:
            pred_weighted_sum += float(pred_mae) * float(pred_count)
            pred_weight_total += int(pred_count)
        if target_mae is not None and target_count > 0:
            target_weighted_sum += float(target_mae) * float(target_count)
            target_weight_total += int(target_count)

        mae_gap = None
        if pred_mae is not None and target_mae is not None:
            mae_gap = float(target_mae) - float(pred_mae)

        rows.append(
            {
                'iteration_name': str(row.get('iteration_name', 'unknown_fold')),
                'pred_vs_ref_count': int(pred_count) if pred_count > 0 else None,
                'pred_vs_ref_mae': pred_mae,
                'pred_vs_ref_std_ae': pred_std,
                'target_vs_ref_count': int(target_count) if target_count > 0 else None,
                'target_vs_ref_mae': target_mae,
                'target_vs_ref_std_ae': target_std,
                'mae_gap_target_minus_pred': mae_gap,
            }
        )

    pred_weighted_mae = (
        float(pred_weighted_sum) / float(pred_weight_total)
        if pred_weight_total > 0
        else None
    )
    target_weighted_mae = (
        float(target_weighted_sum) / float(target_weight_total)
        if target_weight_total > 0
        else None
    )
    combined_gap = None
    if pred_weighted_mae is not None and target_weighted_mae is not None:
        combined_gap = float(target_weighted_mae) - float(pred_weighted_mae)

    rows.append(
        {
            'iteration_name': '__weighted_mean__',
            'pred_vs_ref_count': int(pred_weight_total) if pred_weight_total > 0 else None,
            'pred_vs_ref_mae': pred_weighted_mae,
            'pred_vs_ref_std_ae': None,
            'target_vs_ref_count': int(target_weight_total) if target_weight_total > 0 else None,
            'target_vs_ref_mae': target_weighted_mae,
            'target_vs_ref_std_ae': None,
            'mae_gap_target_minus_pred': combined_gap,
        }
    )

    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f'[analysis] cv_combo error summary written: {out_path}')
    return out_path

def _write_cv_combo_metric_plots(*, cross_fold_summary_path: str, artifacts_root: str) -> dict:
    """Create cross-fold combo plots and dedicated boxplots for validity, uniqueness,
    novelty, and diversity.

    Notes on uncertainty display:
    - We do not draw the same y-error bar on every fold point, because each fold contributes
      only one scalar metric value here, so a per-point uncertainty is not available from this data.
    - Instead, we show:
        1) the raw fold values,
        2) the cross-fold mean,
        3) a mean standard error band,
        4) and a standard deviation band.
    """
    payload = _read_json(cross_fold_summary_path)
    per_fold = list(payload.get('per_fold', []) or [])
    if len(per_fold) == 0:
        print('[analysis] cv_combo skipped: cross-fold summary has no per_fold entries.')
        return {'plot_path': None, 'boxplot_path': None, 'stats_path': None}

    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except Exception:
        print('[analysis] cv_combo skipped: matplotlib is not available in this environment.')
        return {'plot_path': None, 'boxplot_path': None, 'stats_path': None}

    out_dir = os.path.join(artifacts_root, 'cv_combo')
    os.makedirs(out_dir, exist_ok=True)

    metric_specs = [
        ('validity', 'Validity', '#1E3A8A'),
        ('uniqueness', 'Uniqueness', '#0F766E'),
        ('novelty', 'Novelty', '#7C3AED'),
        ('diversity_score', 'Diversity External', '#B45309'),
        ('diversity_internal_score', 'Diversity Internal', '#0EA5E9'),
    ]
    iteration_names = [str(row.get('iteration_name', f'fold_{i}')) for i, row in enumerate(per_fold)]
    x_all = np.arange(len(iteration_names), dtype=float)

    stats_payload = {
        'cross_fold_summary_path': os.path.abspath(cross_fold_summary_path),
        'num_folds': int(len(iteration_names)),
        'metrics': {},
    }

    # -------------------------------------------------------------------------
    # Plot 1: Fold-value line plots + mean + stderr band + std band
    # -------------------------------------------------------------------------
    n_cols = 2
    n_rows = int(np.ceil(float(len(metric_specs)) / float(n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, max(8.5, 4.2 * n_rows)), sharex=True)
    axes_list = list(np.asarray(axes).reshape(-1))

    for ax, (metric_key, metric_title, color) in zip(axes_list, metric_specs):
        values = np.asarray([_to_float(row.get(metric_key)) for row in per_fold], dtype=float)
        valid_mask = np.isfinite(values)
        valid_indices = np.where(valid_mask)[0]
        x = x_all[valid_mask]
        y = values[valid_mask]

        if y.size == 0:
            ax.set_title(f'{metric_title} across folds')
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
            ax.set_ylim(0.0, 1.0)
            stats_payload['metrics'][metric_key] = {
                'count': 0,
                'mean': None,
                'std': None,
                'stderr': None,
                'per_fold': [],
            }
            continue

        mean_val = float(np.mean(y))
        std_val = float(np.std(y, ddof=1)) if y.size > 1 else 0.0
        stderr_val = float(std_val / np.sqrt(y.size)) if y.size > 1 else 0.0

        # Plot raw fold values as a connected line with markers.
        ax.plot(
            x,
            y,
            'o-',
            color=color,
            markersize=5.5,
            linewidth=1.7,
            alpha=0.95,
            label='Fold values',
        )

        # Mean line across the subplot.
        x_min = float(np.min(x)) - 0.4
        x_max = float(np.max(x)) + 0.4
        ax.hlines(
            y=mean_val,
            xmin=x_min,
            xmax=x_max,
            colors='#111111',
            linestyles='--',
            linewidth=1.3,
            label=f'Mean = {mean_val:.4f}',
        )

        # Standard error band around the mean.
        if stderr_val > 0.0:
            ax.fill_between(
                [x_min, x_max],
                [mean_val - stderr_val, mean_val - stderr_val],
                [mean_val + stderr_val, mean_val + stderr_val],
                color='#111111',
                alpha=0.10,
                label=f'Standard error = {stderr_val:.4f}',
            )

        # Standard deviation band around the mean.
        if std_val > 0.0:
            ax.fill_between(
                [x_min, x_max],
                [mean_val - std_val, mean_val - std_val],
                [mean_val + std_val, mean_val + std_val],
                color=color,
                alpha=0.12,
                label=f'Standard deviation = {std_val:.4f}',
            )

        y_pad = max(0.02, std_val * 1.5 if std_val > 0.0 else 0.02)
        y_low = max(0.0, float(np.min(y) - y_pad))
        y_high = min(1.0, float(np.max(y) + y_pad))
        if y_high <= y_low:
            y_low, y_high = 0.0, 1.0

        ax.set_ylim(y_low, y_high)
        ax.set_title(f'{metric_title} across folds')
        ax.set_ylabel(metric_title)
        ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.35)

        legend_handles: list[object] = [
            Line2D([0], [0], color=color, marker='o', linewidth=1.7, markersize=5.5, label='Fold values'),
            Line2D([0], [0], color='#111111', linestyle='--', linewidth=1.3, label=f'Mean = {mean_val:.4f}'),
        ]
        if stderr_val > 0.0:
            legend_handles.append(
                Patch(facecolor='#111111', edgecolor='none', alpha=0.10, label=f'Standard error = {stderr_val:.4f}')
            )
        if std_val > 0.0:
            legend_handles.append(
                Patch(facecolor=color, edgecolor='none', alpha=0.12, label=f'Standard deviation = {std_val:.4f}')
            )

        ax.legend(handles=legend_handles, loc='best', frameon=True, framealpha=0.9)

        stats_payload['metrics'][metric_key] = {
            'count': int(y.size),
            'mean': mean_val,
            'std': std_val,
            'stderr': stderr_val,
            'per_fold': [
                {
                    'iteration_name': iteration_names[int(idx)],
                    'value': float(values[int(idx)]),
                }
                for idx in valid_indices
            ],
        }

    for ax in axes_list[len(metric_specs):]:
        ax.axis('off')

    bottom_start = (n_rows - 1) * n_cols
    for i, ax in enumerate(axes_list[:len(metric_specs)]):
        if i >= bottom_start:
            ax.set_xlabel('CV iteration')

    for ax in axes_list[:len(metric_specs)]:
        ax.set_xticks(x_all)
        ax.set_xticklabels(iteration_names, rotation=30, ha='right')

    fig.suptitle('Cross-fold metrics (V.U.N + external/internal diversity) with uncertainty bands', fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    plot_path = os.path.abspath(os.path.join(out_dir, 'cv_combo_metrics_summary.png'))
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Plot 2: Dedicated boxplots, one subplot per metric
    # -------------------------------------------------------------------------
    boxplot_path = None
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(12, max(8.5, 4.2 * n_rows)), sharey=False)
    axes2_list = list(np.asarray(axes2).reshape(-1))
    have_boxplot_data = False

    for ax2, (metric_key, metric_title, color) in zip(axes2_list, metric_specs):
        metric_vals = np.asarray([_to_float(row.get(metric_key)) for row in per_fold], dtype=float)
        metric_vals = metric_vals[np.isfinite(metric_vals)]

        if metric_vals.size == 0:
            ax2.set_title(f'{metric_title} boxplot')
            ax2.text(0.5, 0.5, 'No data', transform=ax2.transAxes, ha='center', va='center')
            ax2.set_ylim(0.0, 1.0)
            continue

        have_boxplot_data = True

        # Keep only one visible category label on the x axis.
        # This comes from tick_labels here, so we do not set ax2.set_xlabel(...) later.
        bp = ax2.boxplot(
            [metric_vals],
            tick_labels=[metric_title],
            patch_artist=True,
            widths=0.28,
            showmeans=True,
            meanline=True,
        )

        bp['boxes'][0].set(facecolor=color, alpha=0.24, edgecolor=color, linewidth=1.6)
        bp['medians'][0].set(color='#111111', linewidth=1.6)
        bp['means'][0].set(color='#CC0000', linewidth=1.5, linestyle='--')

        # whiskers are lines from box to most extreme non-outlier points
        for whisker in bp['whiskers']:
            whisker.set(color='#444444', linewidth=1.1)

        # caps are the small horizontal lines at whisker ends
        for cap in bp['caps']:
            cap.set(color='#444444', linewidth=1.1)

        # fliers are outliers
        for flier in bp.get('fliers', []):
            flier.set(
                marker='o',
                markersize=6,
                markerfacecolor=color,
                markeredgecolor='#EEFF00',
                alpha=0.9,
            )

        # Put all points at the exact same x position so they line up vertically.
        ax2.scatter(
            np.full(metric_vals.size, 1.0, dtype=float),
            metric_vals,
            s=38,
            alpha=0.92,
            color=color,
            edgecolor='white',
            linewidth=0.6,
            zorder=3,
        )

        local_std = float(np.std(metric_vals, ddof=1)) if metric_vals.size > 1 else 0.0
        y_pad = max(0.01, local_std * 1.8 if local_std > 0.0 else 0.02)
        y_low = max(0.0, float(np.min(metric_vals) - y_pad))
        y_high = min(1.0, float(np.max(metric_vals) + y_pad))
        if y_high <= y_low:
            y_low, y_high = 0.0, 1.0

        ax2.set_ylim(y_low, y_high)
        ax2.set_title(f'{metric_title}-cv-iterations', fontsize=16, fontweight='bold')
        ax2.set_ylabel(metric_title, fontweight='bold', fontsize=15)

        # Do not set xlabel here, because boxplot tick_labels already provide
        # the single category label we want on the x axis.
        # ax2.set_xlabel(metric_title, fontweight='bold', fontsize=15)

        ax2.tick_params(axis='both', which='major', labelsize=15, width=1.5, length=7)
        ax2.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.35)

        legend_handles = [
            Patch(facecolor=color, edgecolor=color, alpha=0.24, label='Interquartile range'),
            Line2D([0], [0], color='#111111', linewidth=1.6, label='Median'),
            Line2D([0], [0], color='#CC0000', linewidth=1.5, linestyle='--', label='Mean'),
            Line2D(
                [0], [0],
                marker='o',
                color='w',
                markerfacecolor=color,
                markeredgecolor='white',
                markersize=7,
                linewidth=0.0,
                label='Fold values',
            ),
        ]
        ax2.legend(handles=legend_handles, loc='best', frameon=True, fontsize=12)

    for ax2 in axes2_list[len(metric_specs):]:
        ax2.axis('off')

    fig2.suptitle('Cross-fold metric boxplots: V.U.N + external/internal diversity', fontsize=16)
    fig2.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    if have_boxplot_data:
        boxplot_path = os.path.abspath(os.path.join(out_dir, 'cv_combo_metrics_boxplots.png'))
        fig2.savefig(boxplot_path, dpi=300)
    plt.close(fig2)

    stats_path = _write_json(os.path.join(out_dir, 'cv_combo_metrics_stats.json'), stats_payload)

    print(f'[analysis] cv_combo plot written: {plot_path}')
    if boxplot_path is not None:
        print(f'[analysis] cv_combo boxplot written: {boxplot_path}')
    print(f'[analysis] cv_combo stats written: {stats_path}')

    return {'plot_path': plot_path, 'boxplot_path': boxplot_path, 'stats_path': stats_path}
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run CV fold iterations: train -> sample -> analysis.')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument('--only-fold', type=int, default=None, help='Run a single CV iteration index only.')
    return parser.parse_args()


def _read_prop_txt_means(path: str) -> list[float]:
    rows = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = str(line).strip().split()
            if len(parts) < 2:
                continue
            try:
                rows.append([float(v) for v in parts[1:]])
            except Exception:
                continue
    if len(rows) == 0:
        raise ValueError(f'Could not compute property means from empty file: {path}')
    mat = np.asarray(rows, dtype=np.float32)
    return np.mean(mat, axis=0).astype(np.float32).tolist()


def _resolve_target_row(cfg: dict, *, train_prop_txt: str, validation_prop_txt: str) -> list[float]:
    mode = str(cfg.get('target_prop_mode', 'mean_test_labels')).strip().lower()
    if mode == 'explicit':
        vals = cfg.get('target_prop')
        if not isinstance(vals, list) or len(vals) == 0:
            raise ValueError("sampling.target_prop_mode='explicit' requires non-empty list sampling.target_prop")
        return [float(v) for v in vals]
    if mode == 'mean_train_labels':
        return _read_prop_txt_means(train_prop_txt)
    if mode == 'mean_test_labels' or mode == 'mean_validation_labels':
        return _read_prop_txt_means(validation_prop_txt)
    raise ValueError("sampling.target_prop_mode must be one of: explicit, mean_train_labels, mean_test_labels")


def _resolve_target_sampling_mode(cfg: dict) -> str:
    """Resolve sampling target mode with legacy-flag compatibility."""
    explicit = cfg.get('target_sampling_mode', None)
    if explicit is not None:
        mode = str(explicit).strip().lower()
    else:
        mode = 'training_dist' if bool(cfg.get('run_training_dist', False)) else 'single_target'

    aliases = {
        'single': 'single_target',
        'single_target': 'single_target',
        'training': 'training_dist',
        'training_dist': 'training_dist',
        'uniform': 'uniform_range',
        'uniform_range': 'uniform_range',
        'uniform_strict': 'uniform_range_strict',
        'uniform_range_strict': 'uniform_range_strict',
    }
    resolved = aliases.get(mode)
    if resolved is None:
        raise ValueError(
            'sampling.target_sampling_mode must be one of: '
            'single_target, training_dist, uniform_range, uniform_range_strict'
        )
    return resolved


def _run_subprocess(command: list[str], *, cwd: str, log_file: Optional[str], step_name: str) -> None:
    print(f'[{step_name}] running command: {command}')
    print(f'[{step_name}] cwd: {cwd}')
    started = time.time()

    env = dict(os.environ)
    env['PYTHONUNBUFFERED'] = '1'

    if not log_file:
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        if proc.stdout is not None:
            for line in proc.stdout:
                print(f'[{step_name}] {line}', end='')
        proc.wait()
        duration = time.time() - started
        print(f'[{step_name}] exit_code={proc.returncode}, duration={duration:.1f}s, log=DISABLED')
        if proc.returncode != 0:
            raise RuntimeError(f'{step_name} failed.')
        return

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f'command: {command}\n')
        f.write(f'cwd: {cwd}\n')
        f.write('streaming: stdout+stderr (tee to console + file)\n')
        f.write('\n=== LIVE OUTPUT ===\n')
        f.flush()

        proc = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        if proc.stdout is not None:
            for line in proc.stdout:
                print(f'[{step_name}] {line}', end='')
                f.write(line)
                f.flush()

        proc.wait()
        duration = time.time() - started
        f.write('\n=== PROCESS SUMMARY ===\n')
        f.write(f'exit_code: {proc.returncode}\n')
        f.write(f'duration_sec: {duration:.3f}\n')

    print(f'[{step_name}] exit_code={proc.returncode}, duration={duration:.1f}s, log={log_file}')
    if proc.returncode != 0:
        raise RuntimeError(f'{step_name} failed. See log: {log_file}')


def _run_analysis_in_process(*, analysis_config_path: str, log_file: Optional[str], step_name: str) -> None:
    started = time.time()

    def _run() -> None:
        cfg = load_analysis_config_from_file(analysis_config_path)
        if bool(cfg.debug):
            print('[analysis:debug] Debug mode enabled from config.')
        summary = run_analysis_pipeline(cfg)
        print(80 * '=')
        print(f'Analysis pipeline finished using config: {analysis_config_path}')
        print(json.dumps(summary, indent=2))

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f'analysis_config: {analysis_config_path}\n')
            f.write('mode: in-process\n\n=== LIVE OUTPUT ===\n')
            f.flush()
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                _run()
        duration = time.time() - started
        print(f'[{step_name}] exit_code=0, duration={duration:.1f}s, log={log_file}')
        return

    _run()
    duration = time.time() - started
    print(f'[{step_name}] exit_code=0, duration={duration:.1f}s, log=DISABLED')


def _build_train_config_for_iteration(
    base_train_config: dict,
    *,
    train_dir: str,
    train_prop_txt: str,
    test_prop_txt: str,
) -> dict:
    override = {
        'data': {
            'prop_file': train_prop_txt,
            'test_prop_file': test_prop_txt,
            # Explicitly ignored by train_labels.py when test_prop_file is provided.
            # Keep a valid value here to make behavior obvious in saved config.
            'train_ratio': 1.0,
        },
        'training': {
            # IMPORTANT: keep model checkpoints + training setup under the training output root
            # (typically under save/, which is gitignored), not under the artifacts output root.
            'save_dir': str(train_dir),
            'use_run_subdir': False,
            'run_name': None,
        },
    }
    return _deep_update_dict(base_train_config, override)


def _print_iteration_assignment(
    *,
    iteration_index: int,
    total_folds: int,
    validation_fold_name: str,
    validation_fold_index: int,
    validation_fold_file: str,
    training_fold_names: list[str],
    training_fold_indices: list[int],
) -> None:
    validation_fold_position = int(validation_fold_index) + 1
    print('')
    print('-' * 90)
    print(
        f'starting CV fold iteration {int(iteration_index)} '
        f'(validation fold parsed from filename: index {validation_fold_position} of {int(total_folds)})'
    )
    print(f'  validation fold name : {validation_fold_name}')
    print(f'  validation fold file : {validation_fold_file}')
    print('  train folds (parsed from filenames):')
    for i, (name, idx) in enumerate(zip(training_fold_names, training_fold_indices)):
        print(f'  {i + 1:>2}. {name} (index={int(idx)})')
    print('-' * 90)
    print('')


def _merge_csv_files_for_analysis(*, csv_paths: list[str], out_csv_path: str) -> str:
    if len(csv_paths) == 0:
        raise ValueError('Cannot merge zero CSV files for analysis train_data_path.')
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError('pandas is required to merge fold training CSV files for analysis.') from e

    frames = []
    for path in csv_paths:
        frames.append(pd.read_csv(path))

    merged = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    merged.to_csv(out_csv_path, index=False)
    return os.path.abspath(out_csv_path)


def _build_analysis_config_for_iteration(
    *,
    analysis_cfg: dict,
    fold_dir: str,
    train_run_dir: str,
    train_data_csv_path: str,
    validation_csv_path: str,
    generated_csv_path: str,
    prediction_eval_csv_path: Optional[str],
    quality_summary_csv_path: Optional[str],
    has_pred_labels: bool,
    has_pred_eval_labels: bool,
    label_column: str,
    training_was_done_this_run: bool,
) -> dict:
    profile = str(analysis_cfg.get('profile', 'bace_pic50_10k'))
    overrides = dict(analysis_cfg.get('overrides', {}) or {})
    
    # Conditionally enable prediction error and residual plots only if training was done.
    # These plots require ground truth which is only available when training runs.
    should_run_groundtruth_plots = bool(training_was_done_this_run and has_pred_eval_labels)
    
    overrides = _deep_update_dict(
        {
            'train_folder': train_run_dir,
            'train_data_path': train_data_csv_path,
            'validation_data_path': validation_csv_path,
            'generated_data_path': generated_csv_path,
            'prediction_eval_data_path': prediction_eval_csv_path,
            'quality_summary_data_path': quality_summary_csv_path,
            'output_dir': os.path.join(fold_dir, 'analysis'),
            'smiles_column': 'smiles',
            'train_sep': ',',
            'validation_sep': ',',
            'generated_sep': ',',
            'target_property_column': str(label_column),
            'predicted_property_column': f'pred_{label_column}' if has_pred_labels else None,
            'run_prediction_error_plot': should_run_groundtruth_plots,
            'run_residual_plot': should_run_groundtruth_plots,
            'training_was_done_this_run': bool(training_was_done_this_run),
            'debug': True,
        },
        overrides,
    )
    return {'profile': profile, 'overrides': overrides}


def _contains_predicted_column(generated_csv_path: str, label_column: str) -> bool:
    try:
        import pandas as pd

        cols = list(pd.read_csv(generated_csv_path, nrows=1).columns)
    except Exception:
        return False
    return f'pred_{label_column}' in cols


def _resolve_prediction_eval_csv_path(*, train_dir: str, train_cfg_path: str) -> Optional[str]:
    try:
        payload = _read_json(train_cfg_path)
    except Exception:
        return None

    training_cfg = dict(payload.get('training', {}) or {})
    filename = str(training_cfg.get('prediction_eval_filename', 'train_validation_prediction_eval.csv')).strip()
    if filename == '':
        filename = 'train_validation_prediction_eval.csv'
    return os.path.abspath(os.path.join(train_dir, filename))


def _print_iteration_start_summary(
    *,
    fold_name: str,
    converted,
    fold_dir: str,
    train_cfg_path: str,
    train_dir: str,
    sample_dir: str,
    analysis_dir: str,
    logs_dir: str,
    sampling_cfg: dict,
    train_enabled: bool,
    sampling_enabled: bool,
    analysis_enabled: bool,
    train_log_enabled: bool,
    analysis_log_enabled: bool,
    total_folds: int,
    validation_fold_index: int,
) -> None:
    validation_fold_position = int(validation_fold_index) + 1
    expected_generated_csv = os.path.abspath(
        os.path.join(sample_dir, str(sampling_cfg.get('result_filename', 'generated.csv')))
    )
    expected_quality_csv = os.path.abspath(
        os.path.join(sample_dir, str(sampling_cfg.get('quality_summary_filename', 'quality_summary.csv')))
    )
    expected_train_log = (
        os.path.join(logs_dir, 'train.log') if bool(train_log_enabled) else 'DISABLED_BY_CONFIG'
    )
    expected_analysis_log = (
        os.path.join(logs_dir, 'analysis.log') if bool(analysis_log_enabled) else 'DISABLED_BY_CONFIG'
    )

    print('')
    print('-' * 90)
    print(f'[{fold_name}] iteration-start summary')
    print('-' * 90)
    print(f'[{fold_name}] cv.validation_fold_index  : {validation_fold_position} / {int(total_folds)}')
    print(f'[{fold_name}] input.validation_fold_name : {converted.validation_fold_name}')
    print(f'[{fold_name}] input.training_fold_names  : {converted.training_fold_names}')
    print(f'[{fold_name}] input.training_csvs        : {converted.training_csvs}')
    print(f'[{fold_name}] input.validation_csv       : {converted.validation_csv}')
    print(f'[{fold_name}] input.train_prop_txt       : {converted.train_prop_txt}')
    print(f'[{fold_name}] input.validation_prop_txt  : {converted.validation_prop_txt}')
    print(f'[{fold_name}] input.train_rows           : {converted.train_rows}')
    print(f'[{fold_name}] input.validation_rows      : {converted.validation_rows}')
    print(f'[{fold_name}] write.train_config_json    : {train_cfg_path}')
    print(f'[{fold_name}] write.train_run_dir        : {train_dir}')
    print(f'[{fold_name}] write.sampling_dir         : {sample_dir}')
    print(f'[{fold_name}] write.analysis_dir         : {analysis_dir}')
    print(f'[{fold_name}] write.logs_dir             : {logs_dir}')
    print(f'[{fold_name}] expected.generated_csv     : {expected_generated_csv}')
    print(f'[{fold_name}] expected.quality_summary   : {expected_quality_csv}')
    print(f'[{fold_name}] expected.train_log         : {expected_train_log}')
    print(f'[{fold_name}] expected.analysis_log      : {expected_analysis_log}')
    print(f'[{fold_name}] train.enabled              : {bool(train_enabled)}')
    print(f'[{fold_name}] sampling.enabled           : {bool(sampling_enabled)}')
    print(f'[{fold_name}] analysis.enabled           : {bool(analysis_enabled)}')
    print('-' * 90)
    print('')


def _print_iteration_end_summary(
    *,
    fold_name: str,
    converted,
    train_cfg_path: str,
    train_dir: str,
    sample_dir: str,
    analysis_dir: str,
    logs_dir: str,
    sampling_result,
    analysis_enabled: bool,
    train_log_enabled: bool,
    analysis_log_enabled: bool,
    analysis_config_path: Optional[str],
    analysis_summary_path: Optional[str],
    iteration_manifest_path: str,
) -> None:
    output_train_log = os.path.join(logs_dir, 'train.log') if bool(train_log_enabled) else 'DISABLED_BY_CONFIG'
    output_analysis_log = (
        os.path.join(logs_dir, 'analysis.log') if bool(analysis_log_enabled) else 'DISABLED_BY_CONFIG'
    )
    print('')
    print('=' * 90)
    print(f'[{fold_name}] iteration-end summary')
    print('=' * 90)
    print(f'[{fold_name}] split.train_rows            : {converted.train_rows}')
    print(f'[{fold_name}] split.validation_rows       : {converted.validation_rows}')
    print(f'[{fold_name}] used.train_config_json      : {train_cfg_path}')
    print(f'[{fold_name}] output.train_run_dir        : {train_dir}')
    print(f'[{fold_name}] output.sampling_dir         : {sample_dir}')
    print(f'[{fold_name}] output.analysis_dir         : {analysis_dir}')
    print(f'[{fold_name}] output.logs_dir             : {logs_dir}')
    print(f'[{fold_name}] output.checkpoint_used      : {sampling_result.checkpoint_path}')
    print(f'[{fold_name}] output.generated_csv        : {sampling_result.generated_csv_path}')
    print(f'[{fold_name}] output.quality_summary_csv  : {sampling_result.quality_summary_csv_path}')
    print(f'[{fold_name}] output.num_generated_saved  : {sampling_result.num_saved}')
    print(f'[{fold_name}] output.train_log            : {output_train_log}')
    print(f'[{fold_name}] output.analysis_log         : {output_analysis_log}')
    print(f'[{fold_name}] output.analysis_enabled     : {bool(analysis_enabled)}')
    print(f'[{fold_name}] output.analysis_config_json : {analysis_config_path}')
    print(f'[{fold_name}] output.analysis_summary_json: {analysis_summary_path}')
    print(f'[{fold_name}] output.iteration_manifest   : {iteration_manifest_path}')
    print('=' * 90)
    print('')


def main() -> None:
    args = _parse_args()
    config_path = _resolve_required_path(args.config, base_dir=os.getcwd(), key_name='--config')
    cfg = _read_json(config_path)

    workspace_root = _resolve_required_path(
        cfg.get('workspace_root', '.'),
        base_dir=_ROOT_DIR,
        key_name='workspace_root',
    )
    if not os.path.isdir(workspace_root):
        raise FileNotFoundError(f'workspace_root does not exist or is not a directory: {workspace_root}')
    os.chdir(workspace_root)

    artifacts_root = _resolve_required_path(
        cfg.get('artifacts_output_root'),
        base_dir=workspace_root,
        key_name='artifacts_output_root',
    )
    training_root = _resolve_required_path(
        cfg.get('training_output_root'),
        base_dir=workspace_root,
        key_name='training_output_root',
    )

    cv_combo_cfg = dict(cfg.get('cv_combo', {}) or {})
    cv_combo_enabled = bool(cv_combo_cfg.get('enabled', True))
    cv_combo_only = bool(cv_combo_cfg.get('only', False))
    cv_combo_summary_override = _resolve_optional_path(
        cv_combo_cfg.get('cross_fold_summary_path'),
        base_dir=workspace_root,
    )

    cleanup_cfg = dict(cfg.get('cleanup', {}) or {})
    purge_artifacts_before_run = bool(cleanup_cfg.get('purge_artifacts_before_run', False))
    purge_training_before_run = bool(cleanup_cfg.get('purge_training_before_run', False))
    purge_python_caches_before_run = bool(cleanup_cfg.get('purge_python_caches_before_run', False))
    remove_iteration_logs_after_success = bool(cleanup_cfg.get('remove_iteration_logs_after_success', False))
    remove_iteration_data_after_success = bool(cleanup_cfg.get('remove_iteration_data_after_success', False))
    remove_sampling_debug_after_success = bool(cleanup_cfg.get('remove_sampling_debug_after_success', False))
    remove_analysis_config_after_success = bool(cleanup_cfg.get('remove_analysis_config_after_success', False))
    remove_iteration_manifest_after_success = bool(
        cleanup_cfg.get('remove_iteration_manifest_after_success', False)
    )
    remove_partial_global_manifest_after_success = bool(
        cleanup_cfg.get('remove_partial_global_manifest_after_success', False)
    )

    manifests_cfg = dict(cfg.get('manifests', {}) or {})
    write_iteration_manifest = bool(manifests_cfg.get('write_iteration_manifest', True))
    write_partial_global_manifest = bool(manifests_cfg.get('write_partial_global_manifest', True))
    write_global_manifest = bool(manifests_cfg.get('write_global_manifest', True))

    train_enabled = bool(cfg.get('train', {}).get('enabled', True))
    sampling_enabled = bool(cfg.get('sampling', {}).get('enabled', True))
    analysis_enabled_global = bool(cfg.get('analysis', {}).get('enabled', True))

    if not train_enabled and sampling_enabled and purge_training_before_run:
        raise ValueError(
            'Invalid configuration: train.enabled=false + sampling.enabled=true + '
            'cleanup.purge_training_before_run=true. This purges checkpoints and then tries to sample. '
            'Set train.enabled=true, or set cleanup.purge_training_before_run=false when reusing existing checkpoints.'
        )

    if not sampling_enabled and analysis_enabled_global and purge_artifacts_before_run:
        raise ValueError(
            'Invalid configuration: sampling.enabled=false + analysis.enabled=true + '
            'cleanup.purge_artifacts_before_run=true. This purges generated inputs required by analysis-only mode. '
            'Set sampling.enabled=true, or set cleanup.purge_artifacts_before_run=false to reuse existing generated CSVs.'
        )

    train_log_enabled = bool(cfg.get('train', {}).get('capture_log_file', True))
    analysis_log_enabled = bool(cfg.get('analysis', {}).get('capture_log_file', True))

    if purge_artifacts_before_run:
        _safe_remove_path(artifacts_root)
    if purge_training_before_run:
        _safe_remove_path(training_root)
    if purge_python_caches_before_run:
        removed_cache_dirs = _purge_python_caches(workspace_root)
        print(f'cleanup: removed __pycache__ directories before run: {removed_cache_dirs}')

    os.makedirs(training_root, exist_ok=True)
    os.makedirs(artifacts_root, exist_ok=True)

    if cv_combo_only:
        # Combo-only mode: skip fold discovery and all per-fold stages.
        cross_fold_summary_path = (
            cv_combo_summary_override
            if cv_combo_summary_override is not None
            else os.path.join(artifacts_root, 'cross_fold_analysis_summary.json')
        )
        cross_fold_summary_path = os.path.abspath(cross_fold_summary_path)
        if not os.path.isfile(cross_fold_summary_path):
            raise FileNotFoundError(
                'cv_combo.only=true but no cross-fold summary was found at '
                f'{cross_fold_summary_path}. Provide cv_combo.cross_fold_summary_path or run analysis first.'
            )

        print('===== CV combo-only mode =====')
        print(f'config_path={config_path}')
        print(f'workspace_root={workspace_root}')
        print(f'artifacts_output_root={artifacts_root}')
        print(f'cross_fold_summary_path={cross_fold_summary_path}')

        cv_combo_outputs = _write_cv_combo_metric_plots(
            cross_fold_summary_path=cross_fold_summary_path,
            artifacts_root=artifacts_root,
        )
        cv_combo_error_summary_path = _write_cv_combo_error_summary_csv(
            cross_fold_summary_path=cross_fold_summary_path,
            artifacts_root=artifacts_root,
        )

        final_manifest_path = os.path.join(artifacts_root, 'global_manifest.json')
        existing_manifest = {}
        if os.path.isfile(final_manifest_path):
            try:
                existing_manifest = _read_json(final_manifest_path)
            except Exception:
                existing_manifest = {}

        existing_manifest['config_path'] = config_path
        existing_manifest['workspace_root'] = workspace_root
        existing_manifest['training_output_root'] = training_root
        existing_manifest['artifacts_output_root'] = artifacts_root
        existing_manifest['cv_combo_only'] = True
        existing_manifest['cross_fold_analysis_summary_path'] = cross_fold_summary_path
        existing_manifest['cv_combo_metrics_plot_path'] = cv_combo_outputs.get('plot_path')
        existing_manifest['cv_combo_metrics_boxplot_path'] = cv_combo_outputs.get('boxplot_path')
        existing_manifest['cv_combo_metrics_stats_path'] = cv_combo_outputs.get('stats_path')
        existing_manifest['cv_combo_error_summary_csv_path'] = cv_combo_error_summary_path
        existing_manifest['finished_unix'] = time.time()
        _write_json(final_manifest_path, existing_manifest)

        print('\n' + '=' * 90)
        print('CV combo-only run completed successfully.')
        print(f'Global manifest: {final_manifest_path}')
        print('=' * 90)
        return

    train_validation_folds_dir = _resolve_required_path(
        cfg.get('train_validation_folds_dir'),
        base_dir=workspace_root,
        key_name='train_validation_folds_dir',
    )

    smiles_column = str(cfg.get('smiles_column', 'smiles'))
    label_columns = [str(x) for x in cfg.get('label_columns', [cfg.get('label_column', 'pIC50')])]
    fold_glob = str(cfg.get('fold_glob', 'fold_*.csv'))

    python_exe = _resolve_python_executable(cfg.get('python_executable'), workspace_root=workspace_root)
    train_script = _resolve_script_path(
        cfg.get('train', {}).get('script', 'train_labels.py'),
        workspace_root=workspace_root,
        key_name='train.script',
    )
    if 'script' in dict(cfg.get('analysis', {}) or {}):
        print('[config] NOTE: analysis.script is ignored; analysis runs in-process via run_fold_pipeline.py.')
    print('===== CV fold iteration pipeline bootstrap =====')
    print(f'config_path={config_path}')
    print(f'workspace_root={workspace_root}')
    print(f'train_validation_folds_dir={train_validation_folds_dir}')
    print(f'fold_glob={fold_glob}')
    print(f'training_output_root={training_root}')
    print(f'artifacts_output_root={artifacts_root}')
    print(f'python_executable={python_exe}')
    print(f'label_columns={label_columns}')
    print(f'stage toggles: train={train_enabled}, sampling={sampling_enabled}, analysis={analysis_enabled_global}')
    print(f'cv_combo toggles: enabled={cv_combo_enabled}, only={cv_combo_only}')
    print(
        'log file toggles: '
        f'train.capture_log_file={train_log_enabled}, '
        f'analysis.capture_log_file={analysis_log_enabled}'
    )
    print(
        'cleanup toggles: '
        f'purge_artifacts_before_run={purge_artifacts_before_run}, '
        f'purge_training_before_run={purge_training_before_run}, '
        f'purge_python_caches_before_run={purge_python_caches_before_run}, '
        f'remove_iteration_logs_after_success={remove_iteration_logs_after_success}, '
        f'remove_iteration_data_after_success={remove_iteration_data_after_success}, '
        f'remove_sampling_debug_after_success={remove_sampling_debug_after_success}, '
        f'remove_analysis_config_after_success={remove_analysis_config_after_success}, '
        f'remove_iteration_manifest_after_success={remove_iteration_manifest_after_success}, '
        f'remove_partial_global_manifest_after_success={remove_partial_global_manifest_after_success}'
    )
    print(
        'manifest toggles: '
        f'write_iteration_manifest={write_iteration_manifest}, '
        f'write_partial_global_manifest={write_partial_global_manifest}, '
        f'write_global_manifest={write_global_manifest}'
    )

    fold_pairs = discover_cv_fold_iterations(
        train_validation_folds_dir=train_validation_folds_dir,
        fold_glob=fold_glob,
    )
    print(f'detected {len(fold_pairs)} folds -> {len(fold_pairs)} CV iterations')

    if args.only_fold is not None:
        fold_pairs = [p for p in fold_pairs if int(p.iteration_index) == int(args.only_fold)]
        if len(fold_pairs) == 0:
            raise ValueError(f'No iteration found for --only-fold={args.only_fold}')
        print(f'filtered to single CV iteration: {args.only_fold}')

    global_manifest = {
        'config_path': config_path,
        'workspace_root': workspace_root,
        'train_validation_folds_dir': train_validation_folds_dir,
        'training_output_root': training_root,
        'artifacts_output_root': artifacts_root,
        'num_iterations': len(fold_pairs),
        'iterations': [],
        'started_unix': time.time(),
    }

    base_train_config = dict(cfg.get('train', {}).get('base_config', {}))
    sampling_cfg = dict(cfg.get('sampling', {}))
    analysis_cfg = dict(cfg.get('analysis', {}))
    heldout_smiles_csv = _resolve_optional_path(sampling_cfg.get('heldout_smiles_csv'), base_dir=workspace_root)

    if heldout_smiles_csv is not None:
        print(f'heldout_smiles_csv={heldout_smiles_csv}')
    else:
        print('heldout_smiles_csv=None')

    total_folds = int(len(fold_pairs))

    for pair in fold_pairs:
        fold_name = f'cv_iteration_{pair.iteration_index}'
        _print_iteration_assignment(
            iteration_index=int(pair.iteration_index),
            total_folds=total_folds,
            validation_fold_name=str(pair.validation_fold.fold_name),
            validation_fold_index=int(pair.validation_fold.fold_index),
            validation_fold_file=os.path.basename(str(pair.validation_fold.csv_path)),
            training_fold_names=[str(f.fold_name) for f in pair.training_folds],
            training_fold_indices=[int(f.fold_index) for f in pair.training_folds],
        )
        # Fold-level manifests/config live with artifacts (not in save/).
        fold_dir = os.path.join(artifacts_root, fold_name)
        training_fold_dir = os.path.join(training_root, fold_name)
        artifacts_iteration_dir = fold_dir
        data_dir = os.path.join(artifacts_iteration_dir, 'data')
        train_dir = os.path.join(training_fold_dir, 'training')
        sample_dir = os.path.join(artifacts_iteration_dir, 'generated')
        analysis_dir = os.path.join(artifacts_iteration_dir, 'analysis')
        logs_dir = os.path.join(artifacts_iteration_dir, 'logs')

        for p in [fold_dir, training_fold_dir, artifacts_iteration_dir, data_dir, train_dir, sample_dir, analysis_dir, logs_dir]:
            os.makedirs(p, exist_ok=True)

        print('\n' + '=' * 90)
        print(f'Running full pipeline for CV iteration {fold_name}')
        print('=' * 90)

        converted = convert_cv_iteration_to_prop_files(
            iteration=pair,
            out_dir=data_dir,
            smiles_column=smiles_column,
            label_columns=label_columns,
        )
        print(
            f'[{fold_name}] data converted: train_rows={converted.train_rows}, '
            f'validation_rows={converted.validation_rows}, '
            f'train_prop_txt={converted.train_prop_txt}, validation_prop_txt={converted.validation_prop_txt}'
        )
        print(f'[{fold_name}] input files: training_csvs={converted.training_csvs}')
        print(f'[{fold_name}] input files: validation_csv={converted.validation_csv}')

        fold_train_cfg = _build_train_config_for_iteration(
            base_train_config,
            train_dir=train_dir,
            train_prop_txt=converted.train_prop_txt,
            test_prop_txt=converted.validation_prop_txt,
        )
        # Training config is part of the model setup; keep it under the training root (save/).
        train_cfg_path = _write_json(os.path.join(training_fold_dir, 'train_config.json'), fold_train_cfg)
        print(f'[{fold_name}] wrote train config: {train_cfg_path}')
        print(
            f'[{fold_name}] split policy: external files only (no random split). '
            f'train_ratio in config is ignored in this mode.'
        )

        run_analysis = bool(analysis_enabled_global)
        _print_iteration_start_summary(
            fold_name=fold_name,
            converted=converted,
            fold_dir=fold_dir,
            train_cfg_path=train_cfg_path,
            train_dir=train_dir,
            sample_dir=sample_dir,
            analysis_dir=analysis_dir,
            logs_dir=logs_dir,
            sampling_cfg=sampling_cfg,
            train_enabled=train_enabled,
            sampling_enabled=sampling_enabled,
            analysis_enabled=run_analysis,
            train_log_enabled=train_log_enabled,
            analysis_log_enabled=analysis_log_enabled,
            total_folds=total_folds,
            validation_fold_index=int(pair.validation_fold.fold_index),
        )

        compact_train_indices = ','.join(str(int(f.fold_index)) for f in pair.training_folds)
        print(
            f'[{fold_name}] split.quick: '
            f'validation={pair.validation_fold.fold_name} '
            f'({int(pair.validation_fold.fold_index) + 1}/{int(total_folds)}) | '
            f'train=[{compact_train_indices}]'
        )

        if train_enabled:
            train_log_path = os.path.join(logs_dir, 'train.log') if bool(train_log_enabled) else None
            _run_subprocess(
                [python_exe, train_script, '--config-json', train_cfg_path],
                cwd=workspace_root,
                log_file=train_log_path,
                step_name=f'{fold_name}:train',
            )
            checkpoint_glob = str(sampling_cfg.get('checkpoint_glob', 'model_best.ckpt-*.pt'))
            _verify_training_stage_outputs(
                fold_name=fold_name,
                train_run_dir=train_dir,
                train_cfg_path=train_cfg_path,
                checkpoint_glob=checkpoint_glob,
                analysis_enabled=bool(run_analysis),
            )
        else:
            print(f'[{fold_name}] training stage disabled by config (train.enabled=False)')

        # Resolve the training run directory used for restore/sampling/analysis.
        # Preferred: save/.../fold_k/training. Back-compat: artifacts/.../fold_k/training.
        train_run_dir_for_sampling = train_dir

        sampling_result = None
        if sampling_enabled:
            fold_sampling_cfg = dict(sampling_cfg)
            # Fallback naming used when checkpoint metadata lacks label_target_names.
            fold_sampling_cfg['pred_property_names'] = list(label_columns)

            train_run_dir_for_sampling = train_dir
            if not train_enabled:
                checkpoint_glob = str(fold_sampling_cfg.get('checkpoint_glob', 'model_best.ckpt-*.pt'))
                resolved_ckpt = _resolve_latest_checkpoint_in_run_dir(
                    run_dir=train_run_dir_for_sampling,
                    checkpoint_glob=checkpoint_glob,
                )
                if resolved_ckpt is None:
                    raise RuntimeError(
                        f'{fold_name}: sampling requested with train.enabled=false, but no checkpoint exists in '
                        f'{train_run_dir_for_sampling} (pattern={checkpoint_glob!r}). '\
                        'Either enable training for this run, or keep existing checkpoints and disable '\
                        'cleanup.purge_training_before_run.'
                    )
                print(f'[{fold_name}] reusing existing checkpoint for sampling: {resolved_ckpt}')

            target_sampling_mode = _resolve_target_sampling_mode(fold_sampling_cfg)
            if target_sampling_mode in ('training_dist', 'uniform_range', 'uniform_range_strict'):
                target_row = None
                print(
                    f'[{fold_name}] sampling mode: {target_sampling_mode} '
                    '(per-molecule varying targets)'
                )
            else:
                target_row = _resolve_target_row(
                    fold_sampling_cfg,
                    train_prop_txt=converted.train_prop_txt,
                    validation_prop_txt=converted.validation_prop_txt,
                )
                print(f'[{fold_name}] sampling target row: {target_row}')

            sampling_result = run_sampling_for_iteration(
                run_dir=train_run_dir_for_sampling,
                output_dir=sample_dir,
                target_row=target_row,
                sampling_cfg=fold_sampling_cfg,
                test_smiles_csv=converted.validation_csv,
                heldout_smiles_csv=heldout_smiles_csv,
                smiles_column=smiles_column,
            )
            print(f'[{fold_name}] sampling complete: saved={sampling_result.num_saved}')
        else:
            print(f'[{fold_name}] sampling stage disabled by config (sampling.enabled=False)')
            expected_generated_csv = os.path.abspath(
                os.path.join(sample_dir, str(sampling_cfg.get('result_filename', 'generated.csv')))
            )
            expected_quality_csv = os.path.abspath(
                os.path.join(sample_dir, str(sampling_cfg.get('quality_summary_filename', 'quality_summary.csv')))
            )
            if run_analysis:
                sampling_result = _resolve_existing_sampling_result_for_analysis_only(
                    fold_name=fold_name,
                    train_run_dir=train_run_dir_for_sampling,
                    generated_csv_path=expected_generated_csv,
                    quality_summary_csv_path=expected_quality_csv,
                )
            else:
                sampling_result = SamplingResult(
                    run_dir=os.path.abspath(train_run_dir_for_sampling),
                    checkpoint_path='SKIPPED_SAMPLING',
                    generated_csv_path=expected_generated_csv,
                    quality_summary_csv_path=expected_quality_csv,
                    num_saved=0,
                    stats={},
                )

        analysis_config_path = None
        analysis_summary_path = None
        if run_analysis:
            if str(sampling_result.generated_csv_path).startswith('SKIPPED_'):
                raise RuntimeError(
                    f'{fold_name}: analysis enabled but generated CSV was not saved. '
                    'Set sampling.save_generated_csv=true or disable analysis.'
                )

            label_for_analysis = label_columns[0] if len(label_columns) > 0 else 'prop_0'
            has_pred_labels = _contains_predicted_column(sampling_result.generated_csv_path, label_for_analysis)
            prediction_eval_csv_path = _resolve_prediction_eval_csv_path(
                train_dir=train_run_dir_for_sampling,
                train_cfg_path=train_cfg_path,
            )
            has_pred_eval_labels = bool(
                prediction_eval_csv_path
                and os.path.isfile(prediction_eval_csv_path)
                and _contains_predicted_column(prediction_eval_csv_path, label_for_analysis)
            )
            merged_train_csv_for_analysis = _merge_csv_files_for_analysis(
                csv_paths=list(converted.training_csvs),
                out_csv_path=os.path.join(data_dir, f'{fold_name}_train_merged.csv'),
            )
            fold_analysis_cfg = _build_analysis_config_for_iteration(
                analysis_cfg=analysis_cfg,
                fold_dir=fold_dir,
                train_run_dir=train_run_dir_for_sampling,
                train_data_csv_path=merged_train_csv_for_analysis,
                validation_csv_path=converted.validation_csv,
                generated_csv_path=sampling_result.generated_csv_path,
                prediction_eval_csv_path=prediction_eval_csv_path,
                quality_summary_csv_path=sampling_result.quality_summary_csv_path,
                has_pred_labels=has_pred_labels,
                has_pred_eval_labels=has_pred_eval_labels,
                label_column=label_for_analysis,
                training_was_done_this_run=train_enabled,
            )
            analysis_config_path = _write_json(os.path.join(analysis_dir, 'analysis_config.json'), fold_analysis_cfg)
            analysis_summary_path = os.path.join(
                analysis_dir,
                str(fold_analysis_cfg.get('overrides', {}).get('summary_json_filename', 'analysis_summary.json')),
            )
            print(f'[{fold_name}] wrote analysis config: {analysis_config_path}')

            _run_analysis_in_process(
                analysis_config_path=analysis_config_path,
                log_file=(os.path.join(logs_dir, 'analysis.log') if bool(analysis_log_enabled) else None),
                step_name=f'{fold_name}:analysis',
            )
        else:
            print(f'[{fold_name}] analysis disabled by config')

        fold_manifest = {
            'iteration_index': int(pair.iteration_index),
            'iteration_name': fold_name,
            'validation_fold_name': converted.validation_fold_name,
            'training_fold_names': converted.training_fold_names,
            'validation_csv': converted.validation_csv,
            'training_csvs': converted.training_csvs,
            'train_prop_txt': converted.train_prop_txt,
            'validation_prop_txt': converted.validation_prop_txt,
            'training_fold_dir': training_fold_dir,
            'artifacts_iteration_dir': artifacts_iteration_dir,
            'train_config_path': train_cfg_path,
            'train_run_dir': train_run_dir_for_sampling if sampling_enabled else train_dir,
            'sampling_result': asdict(sampling_result),
            'analysis_enabled': run_analysis,
            'analysis_config_path': analysis_config_path,
            'analysis_summary_path': analysis_summary_path,
            'completed_unix': time.time(),
        }
        if write_iteration_manifest:
            iteration_manifest_path = _write_json(os.path.join(fold_dir, 'iteration_manifest.json'), fold_manifest)
        else:
            iteration_manifest_path = 'DISABLED_BY_CONFIG'
        global_manifest['iterations'].append(fold_manifest)

        if write_partial_global_manifest:
            _write_json(os.path.join(artifacts_root, 'global_manifest.partial.json'), global_manifest)
        _print_iteration_end_summary(
            fold_name=fold_name,
            converted=converted,
            train_cfg_path=train_cfg_path,
            train_dir=train_run_dir_for_sampling if sampling_enabled else train_dir,
            sample_dir=sample_dir,
            analysis_dir=analysis_dir,
            logs_dir=logs_dir,
            sampling_result=sampling_result,
            analysis_enabled=run_analysis,
            train_log_enabled=train_log_enabled,
            analysis_log_enabled=analysis_log_enabled,
            analysis_config_path=analysis_config_path,
            analysis_summary_path=analysis_summary_path,
            iteration_manifest_path=iteration_manifest_path,
        )

        if remove_sampling_debug_after_success:
            _safe_remove_path(os.path.join(sample_dir, 'sampling_debug.json'))
        if remove_analysis_config_after_success:
            _safe_remove_path(os.path.join(analysis_dir, 'analysis_config.json'))
        if remove_iteration_logs_after_success:
            _safe_remove_path(logs_dir)
        if remove_iteration_data_after_success:
            _safe_remove_path(data_dir)
        if remove_iteration_manifest_after_success and iteration_manifest_path != 'DISABLED_BY_CONFIG':
            _safe_remove_path(iteration_manifest_path)

        print(f'[{fold_name}] CV iteration complete and manifest saved')

    cross_fold_summary_path = None
    cv_combo_outputs = {'plot_path': None, 'boxplot_path': None, 'stats_path': None, 'error_summary_csv_path': None}
    if analysis_enabled_global:
        cross_fold_summary_path = _write_cross_fold_analysis_summary(
            artifacts_root=artifacts_root,
            global_manifest=global_manifest,
        )
        if cross_fold_summary_path is not None and cv_combo_enabled:
            cv_combo_outputs = _write_cv_combo_metric_plots(
                cross_fold_summary_path=cross_fold_summary_path,
                artifacts_root=artifacts_root,
            )
            cv_combo_outputs['error_summary_csv_path'] = _write_cv_combo_error_summary_csv(
                cross_fold_summary_path=cross_fold_summary_path,
                artifacts_root=artifacts_root,
            )
    #This is the second part of the needle
    global_manifest['finished_unix'] = time.time()
    global_manifest['duration_sec'] = float(global_manifest['finished_unix'] - global_manifest['started_unix'])
    global_manifest['cross_fold_analysis_summary_path'] = cross_fold_summary_path
    global_manifest['cv_combo_metrics_plot_path'] = cv_combo_outputs.get('plot_path')
    global_manifest['cv_combo_metrics_boxplot_path'] = cv_combo_outputs.get('boxplot_path')
    global_manifest['cv_combo_metrics_stats_path'] = cv_combo_outputs.get('stats_path')
    global_manifest['cv_combo_error_summary_csv_path'] = cv_combo_outputs.get('error_summary_csv_path')
    if remove_partial_global_manifest_after_success:
        _safe_remove_path(os.path.join(artifacts_root, 'global_manifest.partial.json'))

    if write_global_manifest:
        final_manifest_path = _write_json(os.path.join(artifacts_root, 'global_manifest.json'), global_manifest)
    else:
        final_manifest_path = 'DISABLED_BY_CONFIG'

    print('\n' + '=' * 90)
    print('All requested CV iterations completed successfully.')
    print(f'Global manifest: {final_manifest_path}')
    print('=' * 90)


if __name__ == '__main__':
    main()
