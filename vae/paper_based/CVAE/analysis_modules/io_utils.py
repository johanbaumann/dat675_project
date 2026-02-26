from __future__ import annotations

import json
import os
from typing import Optional

import pandas as pd


def _infer_property_names_from_meta(train_data_path: str) -> list[str]:
    meta_path = f'{train_data_path}.meta.json'
    if not os.path.exists(meta_path):
        return []
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
    except Exception:
        return []

    names = payload.get('property_names') if isinstance(payload, dict) else None
    if not isinstance(names, list):
        return []
    return [str(x) for x in names]


def load_train_dataframe(train_data_path: str, smiles_column: str = 'smiles', sep: Optional[str] = None) -> pd.DataFrame:
    if str(train_data_path).lower().endswith('.csv'):
        df = pd.read_csv(train_data_path, sep=sep)
        if smiles_column not in df.columns:
            raise ValueError(f"Expected smiles column '{smiles_column}' in {train_data_path}")
        return df

    rows = []
    with open(train_data_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 1:
                continue
            rows.append(parts)

    if len(rows) == 0:
        return pd.DataFrame({smiles_column: []})

    prop_names = _infer_property_names_from_meta(train_data_path)
    max_props = max(0, max(len(r) for r in rows) - 1)
    if len(prop_names) != max_props:
        prop_names = [f'prop_{i}' for i in range(max_props)]

    table = {smiles_column: [r[0] for r in rows]}
    for i in range(max_props):
        values = []
        for r in rows:
            if len(r) > i + 1:
                try:
                    values.append(float(r[i + 1]))
                except Exception:
                    values.append(float('nan'))
            else:
                values.append(float('nan'))
        table[prop_names[i]] = values

    return pd.DataFrame(table)


def load_generated_dataframe(generated_data_path: str, sep: str = ',') -> pd.DataFrame:
    return pd.read_csv(generated_data_path, sep=sep)
