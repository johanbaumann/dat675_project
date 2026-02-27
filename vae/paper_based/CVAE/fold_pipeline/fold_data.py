from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from glob import glob
from typing import Iterable

import numpy as np
import pandas as pd


_FOLD_RE = re.compile(r'^fold_iteration_(\d+)\.csv$', re.IGNORECASE)


@dataclass(frozen=True)
class FoldPair:
    fold_index: int
    train_csv: str
    test_csv: str


@dataclass(frozen=True)
class ConvertedFoldData:
    fold_index: int
    train_csv: str
    test_csv: str
    train_prop_txt: str
    test_prop_txt: str
    train_rows: int
    test_rows: int
    train_prop_mean: list[float]
    test_prop_mean: list[float]


def _extract_fold_index(path: str) -> int:
    name = os.path.basename(path)
    m = _FOLD_RE.match(name)
    if not m:
        raise ValueError(f'Expected fold file name like fold_iteration_<idx>.csv, got: {name}')
    return int(m.group(1))


def discover_fold_pairs(
    *,
    train_folds_dir: str,
    test_folds_dir: str,
    fold_glob: str = 'fold_iteration_*.csv',
) -> list[FoldPair]:
    train_paths = sorted(glob(os.path.join(train_folds_dir, fold_glob)))
    test_paths = sorted(glob(os.path.join(test_folds_dir, fold_glob)))

    if len(train_paths) == 0:
        raise FileNotFoundError(f'No train fold files found in: {train_folds_dir} ({fold_glob})')
    if len(test_paths) == 0:
        raise FileNotFoundError(f'No test fold files found in: {test_folds_dir} ({fold_glob})')

    train_map = {_extract_fold_index(p): p for p in train_paths}
    test_map = {_extract_fold_index(p): p for p in test_paths}

    train_idx = set(train_map.keys())
    test_idx = set(test_map.keys())
    if train_idx != test_idx:
        only_train = sorted(train_idx - test_idx)
        only_test = sorted(test_idx - train_idx)
        raise ValueError(
            'Train/test fold index mismatch. '
            f'only_in_train={only_train}, only_in_test={only_test}'
        )

    pairs = [
        FoldPair(
            fold_index=i,
            train_csv=os.path.abspath(train_map[i]),
            test_csv=os.path.abspath(test_map[i]),
        )
        for i in sorted(train_idx)
    ]
    return pairs


def _write_prop_txt_from_csv(
    *,
    csv_path: str,
    out_txt_path: str,
    smiles_column: str,
    label_columns: list[str],
) -> tuple[int, list[float]]:
    df = pd.read_csv(csv_path)
    if smiles_column not in df.columns:
        raise ValueError(f"Missing smiles column '{smiles_column}' in: {csv_path}")

    missing = [c for c in label_columns if c not in df.columns]
    if missing:
        raise ValueError(f'Missing label columns {missing} in: {csv_path}')

    keep = df[[smiles_column] + label_columns].copy()
    keep = keep.dropna(subset=[smiles_column])

    rows = []
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
        'source_csv': os.path.abspath(csv_path),
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


def convert_fold_pair_to_prop_files(
    *,
    pair: FoldPair,
    out_dir: str,
    smiles_column: str,
    label_columns: list[str],
) -> ConvertedFoldData:
    os.makedirs(out_dir, exist_ok=True)
    train_txt = os.path.abspath(os.path.join(out_dir, f'fold_{pair.fold_index}_train_prop.txt'))
    test_txt = os.path.abspath(os.path.join(out_dir, f'fold_{pair.fold_index}_test_prop.txt'))

    train_rows, train_mean = _write_prop_txt_from_csv(
        csv_path=pair.train_csv,
        out_txt_path=train_txt,
        smiles_column=smiles_column,
        label_columns=label_columns,
    )
    test_rows, test_mean = _write_prop_txt_from_csv(
        csv_path=pair.test_csv,
        out_txt_path=test_txt,
        smiles_column=smiles_column,
        label_columns=label_columns,
    )

    converted = ConvertedFoldData(
        fold_index=int(pair.fold_index),
        train_csv=pair.train_csv,
        test_csv=pair.test_csv,
        train_prop_txt=train_txt,
        test_prop_txt=test_txt,
        train_rows=train_rows,
        test_rows=test_rows,
        train_prop_mean=train_mean,
        test_prop_mean=test_mean,
    )

    with open(os.path.join(out_dir, f'fold_{pair.fold_index}_data_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(asdict(converted), f, indent=2)

    return converted


def iter_fold_indexes(pairs: Iterable[FoldPair]) -> list[int]:
    return [int(p.fold_index) for p in pairs]
