from __future__ import annotations

import json
from multiprocessing import Pool
from typing import Optional

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.rdMolDescriptors import CalcNumHBA
from rdkit.Chem.rdMolDescriptors import CalcNumHBD
from rdkit.Chem.rdMolDescriptors import CalcTPSA

#parser = argparse.ArgumentParser()
#parser.add_argument('--input_filename', help='filename for smiles', type=str, default='smiles.txt')
#parser.add_argument('--output_filename', help='name of output file', type=str, default='smiles_prop.txt')
#parser.add_argument('--ncpus', help='number of cpus', type=int, default=1)
#args = parser.parse_args()


args = {
    # Modes:
    # 1) 'rdkit_descriptors': compute descriptors from SMILES (legacy behavior for ZINC experiments)
    # 2) 'from_csv_columns': copy target columns from existing CSV (current BACE workflow)
    # 
    # Set EXACTLY ONE MODE below. The default 'from_csv_columns' is for BACE pIC50 conditioning.
    # For ZINC experiments or others, change mode to 'rdkit_descriptors' and configure 'properties'.
    'mode': 'from_csv_columns',

    # Input/output data files.
    'input_filename': 'bace.csv',
    'output_filename': 'bace_pic50.txt',

    # For mode='rdkit_descriptors': choose descriptor columns to compute.
    # Order defines conditioning order for train/sample.
    # IGNORED if mode='from_csv_columns'
    'properties': ['MW', 'LogP'],  # subset/order of: MW, LogP, TPSA, NumHBD, NumHBA

    # For mode='from_csv_columns': choose source columns from CSV.
    # IGNORED if mode='rdkit_descriptors'
    'smiles_column': 'mol',
    'target_columns': ['pIC50'],

    # Optional descriptor inventory + statistics from the full CSV.
    # Useful for BACE baseline analysis.
    'extract_descriptor_stats': True,
    'descriptor_stats_filename': 'bace_descriptor_stats.csv',
    'descriptor_summary_filename': 'bace_descriptor_summary.json',
    'descriptor_exclude_columns': ['CID', 'Class', 'Model', 'canvasUID'],

    # Parallel workers for rdkit_descriptors mode only.
    # IGNORED if mode='from_csv_columns'
    'ncpus': 1,
}


PROPERTY_FUNCTIONS = {
    'MW': ExactMolWt,
    'LogP': MolLogP,
    'TPSA': CalcTPSA,
    'NumHBD': CalcNumHBD,
    'NumHBA': CalcNumHBA,
}


def _validate_properties(selected: list) -> None:
    if len(selected) == 0:
        raise ValueError('args["properties"] must contain at least one descriptor name.')
    unknown = [p for p in selected if p not in PROPERTY_FUNCTIONS]
    if unknown:
        supported = ', '.join(PROPERTY_FUNCTIONS.keys())
        raise ValueError(f'Unknown property names: {unknown}. Supported: {supported}')

def cal_prop(s: str) -> tuple:
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    props = [PROPERTY_FUNCTIONS[name](m) for name in args['properties']]
    return Chem.MolToSmiles(m), *props


def _write_output_with_metadata(rows: list[tuple], output_filename: str, property_names: list[str], source: str) -> None:
    with open(output_filename, 'w', encoding='utf-8') as w:
        for row in rows:
            if row is None:
                continue
            w.write('\t'.join(map(str, row)) + '\n')

    meta_payload = {
        'property_names': list(property_names),
        'source_file': str(source),
        'num_rows': int(len([r for r in rows if r is not None])),
    }
    with open(output_filename + '.meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta_payload, f, indent=2)


def _extract_descriptor_stats(
    df: pd.DataFrame,
    *,
    smiles_column: str,
    target_columns: list[str],
    exclude_columns: list[str],
    stats_filename: str,
    summary_filename: str,
) -> None:
    excluded = set([smiles_column] + list(target_columns) + list(exclude_columns))

    numeric_cols = [
        c
        for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(numeric_cols) == 0:
        print('No numeric descriptor columns found after exclusions; skipping descriptor stats export.')
        return

    desc = df[numeric_cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    desc['missing_count'] = df[numeric_cols].isna().sum().astype(int)
    desc['missing_rate'] = (desc['missing_count'] / float(len(df))).astype(float)
    desc = desc[
        ['count', 'missing_count', 'missing_rate', 'mean', 'std', 'min', '5%', '25%', '50%', '75%', '95%', 'max']
    ]
    desc = desc.rename(
        columns={
            'count': 'non_null_count',
            '5%': 'p05',
            '25%': 'p25',
            '50%': 'p50',
            '75%': 'p75',
            '95%': 'p95',
        }
    )
    desc.index.name = 'descriptor'
    desc.to_csv(stats_filename)

    summary = {
        'num_rows': int(len(df)),
        'num_total_columns': int(len(df.columns)),
        'num_descriptor_columns': int(len(numeric_cols)),
        'descriptor_columns': list(numeric_cols),
    }
    with open(summary_filename, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('Descriptor statistics extracted from input CSV:')
    print(f"  rows: {summary['num_rows']}")
    print(f"  total columns: {summary['num_total_columns']}")
    print(f"  numeric descriptor columns: {summary['num_descriptor_columns']}")
    print('  first 20 descriptor columns:', summary['descriptor_columns'][:20])
    print(f'  wrote descriptor stats CSV: {stats_filename}')
    print(f'  wrote descriptor summary JSON: {summary_filename}')

def get_file_type(filename: str) -> str:
    if filename.endswith('.smi'):
        return 'smiles'
    elif filename.endswith('.csv'):
        return 'csv'
    elif filename.endswith('txt'):
        return 'txt'
    else:
        raise ValueError('Unsupported file type for input: %s' % filename)

def read_smiles(filename: str) -> list:
    file_type = get_file_type(filename)
    if file_type == 'txt':
        with open(filename) as f:
            smiles = f.read().split('\n')[:-1]
        return smiles
    elif file_type == 'csv':
        df = pd.read_csv(filename)
        if 'smiles' not in df.columns:
            raise ValueError('CSV input file must contain a "smiles" column.')
        return df['smiles'].tolist()


def _prepare_from_csv_columns() -> None:
    input_filename = str(args['input_filename'])
    output_filename = str(args['output_filename'])
    smiles_column = str(args.get('smiles_column', 'smiles'))
    target_columns = [str(c) for c in args.get('target_columns', [])]

    if len(target_columns) == 0:
        raise ValueError('For mode="from_csv_columns", args["target_columns"] must be a non-empty list.')

    df = pd.read_csv(input_filename)
    required = [smiles_column] + target_columns
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns in CSV: {missing}')

    rows: list[tuple] = []
    dropped = 0
    for _, row in df[required].iterrows():
        smiles_raw = row[smiles_column]
        if not isinstance(smiles_raw, str) or smiles_raw.strip() == '':
            dropped += 1
            continue
        mol = Chem.MolFromSmiles(smiles_raw)
        if mol is None:
            dropped += 1
            continue

        target_values: list[float] = []
        bad_target = False
        for col in target_columns:
            value = row[col]
            if pd.isna(value):
                bad_target = True
                break
            try:
                target_values.append(float(value))
            except Exception:
                bad_target = True
                break
        if bad_target:
            dropped += 1
            continue

        rows.append((Chem.MolToSmiles(mol), *target_values))

    print(f'Prepared {len(rows)} rows from CSV columns ({dropped} rows dropped due to invalid smiles/targets).')
    _write_output_with_metadata(
        rows=rows,
        output_filename=output_filename,
        property_names=target_columns,
        source=input_filename,
    )
    print(f'Wrote property file: {output_filename}')
    print(f'Wrote metadata file: {output_filename}.meta.json')

    if bool(args.get('extract_descriptor_stats', False)):
        _extract_descriptor_stats(
            df,
            smiles_column=smiles_column,
            target_columns=target_columns,
            exclude_columns=[str(c) for c in args.get('descriptor_exclude_columns', [])],
            stats_filename=str(args.get('descriptor_stats_filename', 'descriptor_stats.csv')),
            summary_filename=str(args.get('descriptor_summary_filename', 'descriptor_summary.json')),
        )


def _prepare_from_rdkit_descriptors() -> None:
    _validate_properties(args['properties'])
    smiles = read_smiles(args['input_filename'])

    pool = Pool(args['ncpus'])
    print('Calculating properties for %d molecules...' % len(smiles))
    r = pool.map_async(cal_prop, smiles)

    data = r.get()
    pool.close()
    pool.join()

    _write_output_with_metadata(
        rows=data,
        output_filename=str(args['output_filename']),
        property_names=[str(p) for p in args['properties']],
        source=str(args['input_filename']),
    )

    for i, d in enumerate(data):
        if i % 1000 == 0:
            print('Processed %d molecules...' % i)

if __name__ == '__main__':
    mode = str(args.get('mode', 'rdkit_descriptors')).strip().lower()
    if mode == 'rdkit_descriptors':
        _prepare_from_rdkit_descriptors()
    elif mode == 'from_csv_columns':
        _prepare_from_csv_columns()
    else:
        raise ValueError('args["mode"] must be either "rdkit_descriptors" or "from_csv_columns".')

