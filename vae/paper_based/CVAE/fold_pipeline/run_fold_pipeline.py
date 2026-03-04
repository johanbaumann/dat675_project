from __future__ import annotations

import argparse
import json
import os
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


DEFAULT_CONFIG_PATH = os.path.join('fold_pipeline', 'fold_pipeline_config.example.json')


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


def _write_json(path: str, payload: dict) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    return path


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


def _run_subprocess(command: list[str], *, cwd: str, log_file: str, step_name: str) -> None:
    print(f'[{step_name}] running command: {command}')
    print(f'[{step_name}] cwd: {cwd}')
    started = time.time()
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    env = dict(os.environ)
    env['PYTHONUNBUFFERED'] = '1'

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
    print('')
    print('-' * 90)
    print(
        f'starting CV fold iteration {int(iteration_index)} '
        f'(validation fold parsed from filename: index {int(validation_fold_index)} of {int(total_folds)})'
    )
    print(f'  validation fold name : {validation_fold_name}')
    print(f'  validation fold file : {validation_fold_file}')
    print('  train folds (parsed from filenames):')
    for i, (name, idx) in enumerate(zip(training_fold_names, training_fold_indices)):
        print(f'  {i + 1:>2}. {name} (index={int(idx)})')
    print('-' * 90)
    print('')


def _glob_has_any(path_pattern: str) -> bool:
    try:
        import glob

        return any(os.path.isfile(p) for p in glob.glob(str(path_pattern)))
    except Exception:
        return False


def _resolve_training_run_dir_for_iteration_sampling(
    *,
    expected_train_dir: str,
    artifacts_iteration_dir: str,
    checkpoint_glob: str,
) -> str:
    """Return the run dir that actually contains checkpoints.

    New layout (preferred):
      training_output_root/fold_k/training/

    Back-compat: older configs wrote checkpoints under:
      artifacts_output_root/fold_k/training/
    """
    expected_train_dir = str(expected_train_dir)
    if _glob_has_any(os.path.join(expected_train_dir, checkpoint_glob)) or _glob_has_any(
        os.path.join(expected_train_dir, '*.pt')
    ):
        return expected_train_dir

    legacy_dir = os.path.join(str(artifacts_iteration_dir), 'training')
    if os.path.isdir(legacy_dir) and (
        _glob_has_any(os.path.join(legacy_dir, checkpoint_glob)) or _glob_has_any(os.path.join(legacy_dir, '*.pt'))
    ):
        print(
            f'[sampling] NOTE: no checkpoints found in expected training dir: {expected_train_dir}. '\
            f'Falling back to legacy artifacts training dir: {legacy_dir}'
        )
        return legacy_dir

    return expected_train_dir


def _build_analysis_config_for_iteration(
    *,
    analysis_cfg: dict,
    fold_dir: str,
    train_run_dir: str,
    test_csv_path: str,
    generated_csv_path: str,
    has_pred_labels: bool,
    label_column: str,
) -> dict:
    profile = str(analysis_cfg.get('profile', 'bace_pic50_10k'))
    overrides = dict(analysis_cfg.get('overrides', {}) or {})
    overrides = _deep_update_dict(
        {
            'train_folder': train_run_dir,
            'train_data_path': test_csv_path,
            'generated_data_path': generated_csv_path,
            'output_dir': os.path.join(fold_dir, 'analysis'),
            'smiles_column': 'smiles',
            'train_sep': ',',
            'generated_sep': ',',
            'target_property_column': str(label_column),
            'predicted_property_column': f'pred_{label_column}' if has_pred_labels else None,
            'run_prediction_error_plot': bool(has_pred_labels),
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
    total_folds: int,
    validation_fold_index: int,
) -> None:
    expected_generated_csv = os.path.abspath(
        os.path.join(sample_dir, str(sampling_cfg.get('result_filename', 'generated.csv')))
    )
    expected_quality_csv = os.path.abspath(
        os.path.join(sample_dir, str(sampling_cfg.get('quality_summary_filename', 'quality_summary.csv')))
    )

    print('')
    print('-' * 90)
    print(f'[{fold_name}] iteration-start summary')
    print('-' * 90)
    print(f'[{fold_name}] cv.validation_fold_index  : {int(validation_fold_index)} / {int(total_folds)}')
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
    print(f'[{fold_name}] expected.train_log         : {os.path.join(logs_dir, "train.log")}')
    print(f'[{fold_name}] expected.analysis_log      : {os.path.join(logs_dir, "analysis.log")}')
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
    analysis_config_path: Optional[str],
    iteration_manifest_path: str,
) -> None:
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
    print(f'[{fold_name}] output.train_log            : {os.path.join(logs_dir, "train.log")}')
    print(f'[{fold_name}] output.analysis_log         : {os.path.join(logs_dir, "analysis.log")}')
    print(f'[{fold_name}] output.analysis_enabled     : {bool(analysis_enabled)}')
    print(f'[{fold_name}] output.analysis_config_json : {analysis_config_path}')
    print(f'[{fold_name}] output.iteration_manifest   : {iteration_manifest_path}')
    print('=' * 90)
    print('')


def main() -> None:
    args = _parse_args()
    cfg = _read_json(args.config)

    workspace_root = os.path.abspath(str(cfg.get('workspace_root', _ROOT_DIR)))
    os.chdir(workspace_root)

    folds_dir_raw = cfg.get('train_validation_folds_dir', cfg.get('train_validation_folds'))
    if folds_dir_raw is None:
        raise KeyError('Config is missing required key: train_validation_folds_dir')
    train_validation_folds_dir = os.path.abspath(str(folds_dir_raw))

    # Output roots:
    # - training_output_root: checkpoints/history/training config (ideally under save/)
    # - artifacts_output_root: fold data, sampling outputs, analysis outputs, logs (outside save/)
    # Backward compatibility:
    # - output_root is treated as artifacts_output_root when artifacts_output_root is not provided.
    output_root_cfg = cfg.get('output_root', None)
    artifacts_root_cfg = cfg.get('artifacts_output_root', None)
    training_root_cfg = cfg.get('training_output_root', None)

    if artifacts_root_cfg is None:
        artifacts_root_cfg = output_root_cfg
    if training_root_cfg is None:
        training_root_cfg = output_root_cfg

    if artifacts_root_cfg is None:
        artifacts_root_cfg = os.path.join('fold_pipeline_outputs')
    if training_root_cfg is None:
        training_root_cfg = os.path.join('save', 'fold_pipeline_runs')

    # Keep the printed/output_root key aligned with where fold manifests live.
    output_root = os.path.abspath(str(artifacts_root_cfg))
    training_root = os.path.abspath(str(training_root_cfg))
    artifacts_root = os.path.abspath(str(artifacts_root_cfg))
    smiles_column = str(cfg.get('smiles_column', 'smiles'))
    label_columns = [str(x) for x in cfg.get('label_columns', [cfg.get('label_column', 'pIC50')])]
    fold_glob = str(cfg.get('fold_glob', 'fold_*.csv'))

    python_exe = str(cfg.get('python_executable') or sys.executable)
    train_script = str(cfg.get('train', {}).get('script', 'train_labels.py'))
    analysis_script = str(cfg.get('analysis', {}).get('script', 'run_viz_pipeline.py'))
    train_enabled = bool(cfg.get('train', {}).get('enabled', True))
    sampling_enabled = bool(cfg.get('sampling', {}).get('enabled', True))
    analysis_enabled_global = bool(cfg.get('analysis', {}).get('enabled', True))

    print('===== CV fold iteration pipeline bootstrap =====')
    print(f'workspace_root={workspace_root}')
    print(f'train_validation_folds_dir={train_validation_folds_dir}')
    print(f'fold_glob={fold_glob}')
    print(f'output_root={output_root}')
    print(f'training_output_root={training_root}')
    print(f'artifacts_output_root={artifacts_root}')
    print(f'python_executable={python_exe}')
    print(f'label_columns={label_columns}')
    print(f'stage toggles: train={train_enabled}, sampling={sampling_enabled}, analysis={analysis_enabled_global}')

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

    os.makedirs(output_root, exist_ok=True)
    os.makedirs(training_root, exist_ok=True)
    os.makedirs(artifacts_root, exist_ok=True)

    global_manifest = {
        'config_path': os.path.abspath(args.config),
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
            total_folds=total_folds,
            validation_fold_index=int(pair.validation_fold.fold_index),
        )

        compact_train_indices = ','.join(str(int(f.fold_index)) for f in pair.training_folds)
        print(
            f'[{fold_name}] split.quick: '
            f'validation={pair.validation_fold.fold_name} '
            f'({int(pair.validation_fold.fold_index)}/{int(total_folds)}) | '
            f'train=[{compact_train_indices}]'
        )

        if train_enabled:
            _run_subprocess(
                [python_exe, train_script, '--config-json', train_cfg_path],
                cwd=workspace_root,
                log_file=os.path.join(logs_dir, 'train.log'),
                step_name=f'{fold_name}:train',
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

            checkpoint_glob = str(fold_sampling_cfg.get('checkpoint_glob', 'model_best.ckpt-*.pt'))
            train_run_dir_for_sampling = _resolve_training_run_dir_for_iteration_sampling(
                expected_train_dir=train_dir,
                artifacts_iteration_dir=artifacts_iteration_dir,
                checkpoint_glob=checkpoint_glob,
            )

            run_training_dist = bool(fold_sampling_cfg.get('run_training_dist', False))
            if run_training_dist:
                target_row = None
                print(f'[{fold_name}] sampling mode: training_dist (per-molecule varying targets)')
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
            sampling_result = SamplingResult(
                run_dir=os.path.abspath(train_run_dir_for_sampling),
                checkpoint_path='SKIPPED_SAMPLING',
                generated_csv_path=expected_generated_csv,
                quality_summary_csv_path=expected_quality_csv,
                num_saved=0,
                stats={},
            )

        analysis_config_path = None
        if run_analysis:
            if str(sampling_result.checkpoint_path).startswith('SKIPPED_'):
                raise RuntimeError(
                    f'{fold_name}: analysis enabled but sampling was disabled. '
                    'Set analysis.enabled=false or enable sampling.'
                )
            if str(sampling_result.generated_csv_path).startswith('SKIPPED_'):
                raise RuntimeError(
                    f'{fold_name}: analysis enabled but generated CSV was not saved. '
                    'Set sampling.save_generated_csv=true or disable analysis.'
                )

            label_for_analysis = label_columns[0] if len(label_columns) > 0 else 'prop_0'
            has_pred_labels = _contains_predicted_column(sampling_result.generated_csv_path, label_for_analysis)
            fold_analysis_cfg = _build_analysis_config_for_iteration(
                analysis_cfg=analysis_cfg,
                fold_dir=fold_dir,
                train_run_dir=train_run_dir_for_sampling,
                test_csv_path=converted.validation_csv,
                generated_csv_path=sampling_result.generated_csv_path,
                has_pred_labels=has_pred_labels,
                label_column=label_for_analysis,
            )
            analysis_config_path = _write_json(os.path.join(analysis_dir, 'analysis_config.json'), fold_analysis_cfg)
            print(f'[{fold_name}] wrote analysis config: {analysis_config_path}')

            _run_subprocess(
                [python_exe, analysis_script, '--config', analysis_config_path],
                cwd=workspace_root,
                log_file=os.path.join(logs_dir, 'analysis.log'),
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
            'completed_unix': time.time(),
        }
        iteration_manifest_path = _write_json(os.path.join(fold_dir, 'iteration_manifest.json'), fold_manifest)
        global_manifest['iterations'].append(fold_manifest)

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
            analysis_config_path=analysis_config_path,
            iteration_manifest_path=iteration_manifest_path,
        )
        print(f'[{fold_name}] CV iteration complete and manifest saved')

    global_manifest['finished_unix'] = time.time()
    global_manifest['duration_sec'] = float(global_manifest['finished_unix'] - global_manifest['started_unix'])
    final_manifest_path = _write_json(os.path.join(artifacts_root, 'global_manifest.json'), global_manifest)

    print('\n' + '=' * 90)
    print('All requested CV iterations completed successfully.')
    print(f'Global manifest: {final_manifest_path}')
    print('=' * 90)


if __name__ == '__main__':
    main()
