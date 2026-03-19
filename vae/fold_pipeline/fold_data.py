from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from glob import glob
from typing import Iterable

import numpy as np
import pandas as pd


_FOLD_TRAILING_INT_RE = re.compile(r'^(.*?)(\d+)$')


@dataclass(frozen=True)
class FoldFile:
    fold_index: int
    fold_name: str
    csv_path: str


@dataclass(frozen=True)
class CVFoldIteration:
    iteration_index: int
    validation_fold: FoldFile
    training_folds: list[FoldFile]


@dataclass(frozen=True)
class ConvertedCVIterationData:
    iteration_index: int
    iteration_name: str
    validation_fold_name: str
    training_fold_names: list[str]
    validation_csv: str
    training_csvs: list[str]
    train_prop_txt: str
    validation_prop_txt: str
    train_rows: int
    validation_rows: int
    train_prop_mean: list[float]
    validation_prop_mean: list[float]


def _extract_fold_index(path: str) -> tuple[int, str]:
    stem = os.path.splitext(os.path.basename(path))[0]
    m = _FOLD_TRAILING_INT_RE.match(stem)
    if not m:
        raise ValueError(
            f'Could not extract trailing fold index from filename: {os.path.basename(path)}. '
            'Expected names ending with an integer (e.g., fold_0.csv, fold_iteration_3.csv).'
        )
    idx = int(m.group(2))
    return idx, stem


def discover_cv_fold_iterations(
    *,
    train_validation_folds_dir: str,
    fold_glob: str = 'fold_*.csv',
) -> list[CVFoldIteration]:
    fold_paths = sorted(glob(os.path.join(train_validation_folds_dir, fold_glob)))
    if len(fold_paths) == 0:
        raise FileNotFoundError(
            f'No fold files found in: {train_validation_folds_dir} ({fold_glob})'
        )
    if len(fold_paths) < 2:
        raise ValueError('At least 2 fold CSV files are required for CV iterations.')

    fold_files: list[FoldFile] = []
    seen_idx: set[int] = set()
    for path in fold_paths:
        idx, fold_name = _extract_fold_index(path)
        if idx in seen_idx:
            raise ValueError(f'Duplicate fold index detected for index={idx}. File={path}')
        seen_idx.add(idx)
        fold_files.append(
            FoldFile(
                fold_index=idx,
                fold_name=fold_name,
                csv_path=os.path.abspath(path),
            )
        )

    fold_files = sorted(fold_files, key=lambda x: int(x.fold_index))
    iterations: list[CVFoldIteration] = []
    for i, validation_fold in enumerate(fold_files):
        train_folds = [f for j, f in enumerate(fold_files) if j != i]
        iterations.append(
            CVFoldIteration(
                iteration_index=i,
                validation_fold=validation_fold,
                training_folds=train_folds,
            )
        )
    return iterations


def _write_prop_txt_from_csvs(
    *,
    csv_paths: list[str],
    out_txt_path: str,
    smiles_column: str,
    label_columns: list[str],
) -> tuple[int, list[float]]:
    if len(csv_paths) == 0:
        raise ValueError('csv_paths cannot be empty')

    rows = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if smiles_column not in df.columns:
            raise ValueError(f"Missing smiles column '{smiles_column}' in: {csv_path}")

        missing = [c for c in label_columns if c not in df.columns]
        if missing:
            raise ValueError(f'Missing label columns {missing} in: {csv_path}')

        keep = df[[smiles_column] + label_columns].copy()
        keep = keep.dropna(subset=[smiles_column])

        for _, row in keep.iterrows():
            smi = str(row[smiles_column]).strip()
            if not smi:
                continue
            vals = []
            bad = False
            for col in label_columns:
                try:
                    vals.append(float(row[col]))
                except Exception:
                    bad = True
                    break
            if bad:
                continue
            rows.append((smi, vals))

    os.makedirs(os.path.dirname(out_txt_path), exist_ok=True)
    with open(out_txt_path, 'w', encoding='utf-8') as f:
        for smi, vals in rows:
            right = ' '.join(f'{v:.10g}' for v in vals)
            f.write(f'{smi} {right}\n')

    # Persist sidecar metadata so downstream training/sampling can recover
    # human-readable property names (e.g., pIC50) instead of falling back to
    # generic names like prop_0.
    meta_payload = {
        'source_csvs': [os.path.abspath(p) for p in csv_paths],
        'smiles_column': str(smiles_column),
        'property_names': [str(c) for c in label_columns],
        'num_rows': int(len(rows)),
    }
    with open(f'{out_txt_path}.meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta_payload, f, indent=2)

    if len(rows) == 0:
        raise ValueError(f'No valid rows written to property txt: {out_txt_path}')

    prop_matrix = np.asarray([vals for _, vals in rows], dtype=np.float32)
    prop_mean = np.mean(prop_matrix, axis=0).astype(np.float32).tolist()
    return int(len(rows)), [float(v) for v in prop_mean]


def convert_cv_iteration_to_prop_files(
    *,
    iteration: CVFoldIteration,
    out_dir: str,
    smiles_column: str,
    label_columns: list[str],
) -> ConvertedCVIterationData:
    os.makedirs(out_dir, exist_ok=True)
    train_txt = os.path.abspath(os.path.join(out_dir, f'cv_iteration_{iteration.iteration_index}_train_prop.txt'))
    val_txt = os.path.abspath(
        os.path.join(out_dir, f'cv_iteration_{iteration.iteration_index}_validation_prop.txt')
    )

    train_csvs = [str(f.csv_path) for f in iteration.training_folds]
    validation_csv = str(iteration.validation_fold.csv_path)

    train_rows, train_mean = _write_prop_txt_from_csvs(
        csv_paths=train_csvs,
        out_txt_path=train_txt,
        smiles_column=smiles_column,
        label_columns=label_columns,
    )
    validation_rows, validation_mean = _write_prop_txt_from_csvs(
        csv_paths=[validation_csv],
        out_txt_path=val_txt,
        smiles_column=smiles_column,
        label_columns=label_columns,
    )

    converted = ConvertedCVIterationData(
        iteration_index=int(iteration.iteration_index),
        iteration_name=f'cv_iteration_{int(iteration.iteration_index)}',
        validation_fold_name=str(iteration.validation_fold.fold_name),
        training_fold_names=[str(f.fold_name) for f in iteration.training_folds],
        validation_csv=validation_csv,
        training_csvs=train_csvs,
        train_prop_txt=train_txt,
        validation_prop_txt=val_txt,
        train_rows=train_rows,
        validation_rows=validation_rows,
        train_prop_mean=train_mean,
        validation_prop_mean=validation_mean,
    )

    with open(
        os.path.join(out_dir, f'cv_iteration_{iteration.iteration_index}_data_manifest.json'),
        'w',
        encoding='utf-8',
    ) as f:
        json.dump(asdict(converted), f, indent=2)

    return converted


def iter_fold_indexes(iterations: Iterable[CVFoldIteration]) -> list[int]:
    return [int(p.validation_fold.fold_index) for p in iterations]
