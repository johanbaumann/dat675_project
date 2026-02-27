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

from fold_pipeline.fold_data import convert_fold_pair_to_prop_files, discover_fold_pairs
from fold_pipeline.sampling_pipeline import run_sampling_for_fold


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


def _write_json(path: str, payload: dict) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run full fold pipeline: train -> sample -> analysis.')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument('--only-fold', type=int, default=None, help='Run a single fold index only.')
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


def _resolve_target_row(cfg: dict, *, train_prop_txt: str, test_prop_txt: str) -> list[float]:
    mode = str(cfg.get('target_prop_mode', 'mean_test_labels')).strip().lower()
    if mode == 'explicit':
        vals = cfg.get('target_prop')
        if not isinstance(vals, list) or len(vals) == 0:
            raise ValueError("sampling.target_prop_mode='explicit' requires non-empty list sampling.target_prop")
        return [float(v) for v in vals]
    if mode == 'mean_train_labels':
        return _read_prop_txt_means(train_prop_txt)
    if mode == 'mean_test_labels':
        return _read_prop_txt_means(test_prop_txt)
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


def _build_train_config_for_fold(base_train_config: dict, *, fold_dir: str, train_prop_txt: str, test_prop_txt: str) -> dict:
    override = {
        'data': {
            'prop_file': train_prop_txt,
            'test_prop_file': test_prop_txt,
            # Explicitly ignored by train_labels.py when test_prop_file is provided.
            # Keep a valid value here to make behavior obvious in saved config.
            'train_ratio': 1.0,
        },
        'training': {
            'save_dir': os.path.join(fold_dir, 'training'),
            'use_run_subdir': False,
            'run_name': None,
        },
    }
    return _deep_update_dict(base_train_config, override)


def _build_analysis_config_for_fold(
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


def _print_fold_start_summary(
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
    analysis_enabled: bool,
) -> None:
    expected_generated_csv = os.path.abspath(
        os.path.join(sample_dir, str(sampling_cfg.get('result_filename', 'generated.csv')))
    )
    expected_quality_csv = os.path.abspath(
        os.path.join(sample_dir, str(sampling_cfg.get('quality_summary_filename', 'quality_summary.csv')))
    )

    print('')
    print('-' * 90)
    print(f'[{fold_name}] fold-start summary')
    print('-' * 90)
    print(f'[{fold_name}] input.train_csv            : {converted.train_csv}')
    print(f'[{fold_name}] input.test_csv             : {converted.test_csv}')
    print(f'[{fold_name}] input.train_prop_txt       : {converted.train_prop_txt}')
    print(f'[{fold_name}] input.test_prop_txt        : {converted.test_prop_txt}')
    print(f'[{fold_name}] input.train_rows           : {converted.train_rows}')
    print(f'[{fold_name}] input.test_rows            : {converted.test_rows}')
    print(f'[{fold_name}] write.train_config_json    : {train_cfg_path}')
    print(f'[{fold_name}] write.train_run_dir        : {train_dir}')
    print(f'[{fold_name}] write.sampling_dir         : {sample_dir}')
    print(f'[{fold_name}] write.analysis_dir         : {analysis_dir}')
    print(f'[{fold_name}] write.logs_dir             : {logs_dir}')
    print(f'[{fold_name}] expected.generated_csv     : {expected_generated_csv}')
    print(f'[{fold_name}] expected.quality_summary   : {expected_quality_csv}')
    print(f'[{fold_name}] expected.train_log         : {os.path.join(logs_dir, "train.log")}')
    print(f'[{fold_name}] expected.analysis_log      : {os.path.join(logs_dir, "analysis.log")}')
    print(f'[{fold_name}] analysis.enabled           : {bool(analysis_enabled)}')
    print('-' * 90)
    print('')


def _print_fold_end_summary(
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
    fold_manifest_path: str,
) -> None:
    print('')
    print('=' * 90)
    print(f'[{fold_name}] fold-end summary')
    print('=' * 90)
    print(f'[{fold_name}] split.train_rows            : {converted.train_rows}')
    print(f'[{fold_name}] split.test_rows             : {converted.test_rows}')
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
    print(f'[{fold_name}] output.fold_manifest_json   : {fold_manifest_path}')
    print('=' * 90)
    print('')


def main() -> None:
    args = _parse_args()
    cfg = _read_json(args.config)

    workspace_root = os.path.abspath(str(cfg.get('workspace_root', _ROOT_DIR)))
    os.chdir(workspace_root)

    train_folds_dir = os.path.abspath(str(cfg['train_folds_dir']))
    test_folds_dir = os.path.abspath(str(cfg['test_folds_dir']))
    output_root = os.path.abspath(str(cfg.get('output_root', os.path.join('save', 'fold_pipeline_runs'))))
    smiles_column = str(cfg.get('smiles_column', 'smiles'))
    label_columns = [str(x) for x in cfg.get('label_columns', [cfg.get('label_column', 'pIC50')])]
    fold_glob = str(cfg.get('fold_glob', 'fold_iteration_*.csv'))

    python_exe = str(cfg.get('python_executable') or sys.executable)
    train_script = str(cfg.get('train', {}).get('script', 'train_labels.py'))
    analysis_script = str(cfg.get('analysis', {}).get('script', 'run_viz_pipeline.py'))

    print('===== fold pipeline bootstrap =====')
    print(f'workspace_root={workspace_root}')
    print(f'train_folds_dir={train_folds_dir}')
    print(f'test_folds_dir={test_folds_dir}')
    print(f'output_root={output_root}')
    print(f'python_executable={python_exe}')
    print(f'label_columns={label_columns}')

    fold_pairs = discover_fold_pairs(
        train_folds_dir=train_folds_dir,
        test_folds_dir=test_folds_dir,
        fold_glob=fold_glob,
    )
    print(f'detected {len(fold_pairs)} fold pairs based on train/test folders')

    if args.only_fold is not None:
        fold_pairs = [p for p in fold_pairs if int(p.fold_index) == int(args.only_fold)]
        if len(fold_pairs) == 0:
            raise ValueError(f'No fold found for --only-fold={args.only_fold}')
        print(f'filtered to single fold: {args.only_fold}')

    os.makedirs(output_root, exist_ok=True)

    global_manifest = {
        'config_path': os.path.abspath(args.config),
        'workspace_root': workspace_root,
        'train_folds_dir': train_folds_dir,
        'test_folds_dir': test_folds_dir,
        'num_folds': len(fold_pairs),
        'folds': [],
        'started_unix': time.time(),
    }

    base_train_config = dict(cfg.get('train', {}).get('base_config', {}))
    sampling_cfg = dict(cfg.get('sampling', {}))
    analysis_cfg = dict(cfg.get('analysis', {}))

    for pair in fold_pairs:
        fold_name = f'fold_{pair.fold_index}'
        fold_dir = os.path.join(output_root, fold_name)
        data_dir = os.path.join(fold_dir, 'data')
        train_dir = os.path.join(fold_dir, 'training')
        sample_dir = os.path.join(fold_dir, 'sampling')
        analysis_dir = os.path.join(fold_dir, 'analysis')
        logs_dir = os.path.join(fold_dir, 'logs')

        for p in [fold_dir, data_dir, train_dir, sample_dir, analysis_dir, logs_dir]:
            os.makedirs(p, exist_ok=True)

        print('\n' + '=' * 90)
        print(f'Running full pipeline for {fold_name}')
        print('=' * 90)

        converted = convert_fold_pair_to_prop_files(
            pair=pair,
            out_dir=data_dir,
            smiles_column=smiles_column,
            label_columns=label_columns,
        )
        print(
            f'[{fold_name}] data converted: train_rows={converted.train_rows}, test_rows={converted.test_rows}, '
            f'train_prop_txt={converted.train_prop_txt}, test_prop_txt={converted.test_prop_txt}'
        )
        print(f'[{fold_name}] input files: train_csv={converted.train_csv}')
        print(f'[{fold_name}] input files: test_csv={converted.test_csv}')

        fold_train_cfg = _build_train_config_for_fold(
            base_train_config,
            fold_dir=fold_dir,
            train_prop_txt=converted.train_prop_txt,
            test_prop_txt=converted.test_prop_txt,
        )
        train_cfg_path = _write_json(os.path.join(fold_dir, 'train_config.json'), fold_train_cfg)
        print(f'[{fold_name}] wrote train config: {train_cfg_path}')
        print(
            f'[{fold_name}] split policy: external files only (no random split). '
            f'train_ratio in config is ignored in this mode.'
        )

        run_analysis = bool(analysis_cfg.get('enabled', True))
        _print_fold_start_summary(
            fold_name=fold_name,
            converted=converted,
            fold_dir=fold_dir,
            train_cfg_path=train_cfg_path,
            train_dir=train_dir,
            sample_dir=sample_dir,
            analysis_dir=analysis_dir,
            logs_dir=logs_dir,
            sampling_cfg=sampling_cfg,
            analysis_enabled=run_analysis,
        )

        _run_subprocess(
            [python_exe, train_script, '--config-json', train_cfg_path],
            cwd=workspace_root,
            log_file=os.path.join(logs_dir, 'train.log'),
            step_name=f'{fold_name}:train',
        )

        target_row = _resolve_target_row(
            sampling_cfg,
            train_prop_txt=converted.train_prop_txt,
            test_prop_txt=converted.test_prop_txt,
        )
        print(f'[{fold_name}] sampling target row: {target_row}')

        sampling_result = run_sampling_for_fold(
            run_dir=train_dir,
            output_dir=sample_dir,
            target_row=target_row,
            sampling_cfg=sampling_cfg,
        )
        print(f'[{fold_name}] sampling complete: saved={sampling_result.num_saved}')

        analysis_config_path = None
        if run_analysis:
            label_for_analysis = label_columns[0] if len(label_columns) > 0 else 'prop_0'
            has_pred_labels = _contains_predicted_column(sampling_result.generated_csv_path, label_for_analysis)
            fold_analysis_cfg = _build_analysis_config_for_fold(
                analysis_cfg=analysis_cfg,
                fold_dir=fold_dir,
                train_run_dir=train_dir,
                test_csv_path=converted.test_csv,
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
            'fold_index': int(pair.fold_index),
            'fold_name': fold_name,
            'train_csv': converted.train_csv,
            'test_csv': converted.test_csv,
            'train_prop_txt': converted.train_prop_txt,
            'test_prop_txt': converted.test_prop_txt,
            'train_config_path': train_cfg_path,
            'train_run_dir': train_dir,
            'sampling_result': asdict(sampling_result),
            'analysis_enabled': run_analysis,
            'analysis_config_path': analysis_config_path,
            'completed_unix': time.time(),
        }
        fold_manifest_path = _write_json(os.path.join(fold_dir, 'fold_manifest.json'), fold_manifest)
        global_manifest['folds'].append(fold_manifest)

        _write_json(os.path.join(output_root, 'global_manifest.partial.json'), global_manifest)
        _print_fold_end_summary(
            fold_name=fold_name,
            converted=converted,
            train_cfg_path=train_cfg_path,
            train_dir=train_dir,
            sample_dir=sample_dir,
            analysis_dir=analysis_dir,
            logs_dir=logs_dir,
            sampling_result=sampling_result,
            analysis_enabled=run_analysis,
            analysis_config_path=analysis_config_path,
            fold_manifest_path=fold_manifest_path,
        )
        print(f'[{fold_name}] fold complete and manifest saved')

    global_manifest['finished_unix'] = time.time()
    global_manifest['duration_sec'] = float(global_manifest['finished_unix'] - global_manifest['started_unix'])
    final_manifest_path = _write_json(os.path.join(output_root, 'global_manifest.json'), global_manifest)

    print('\n' + '=' * 90)
    print('All requested folds completed successfully.')
    print(f'Global manifest: {final_manifest_path}')
    print('=' * 90)


if __name__ == '__main__':
    main()
