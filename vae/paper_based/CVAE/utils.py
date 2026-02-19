#import h5py
import numpy as np
from rdkit import Chem
from typing import Any, Callable, Optional
import json
import os
import gzip
import pickle
import glob
import re
import time

try:
    from rdkit.Chem.MolStandardize import rdMolStandardize
except Exception:
    rdMolStandardize = None


TRAIN_CONFIG_DEFAULTS = {
    'training_preset': 'custom',
    'batch_size': 128,
    'latent_size': 200,
    'unit_size': 512,
    'n_rnn_layer': 3,
    'seq_length': 120,
    'prop_file': 'smiles_prop.txt',
    'mean': 0.0,
    'stddev': 1.0,
    'num_epochs': 100,
    'lr': 0.0001,
    'num_prop': 3,
    'grad_clip_norm': 1.0,
    'save_dir': 'save/',
    'run_name': None,
    'use_run_subdir': True,
    'patientce': 10,
    'model_mode': 'lstm',
    'optimizer': 'adam',
    'weight_decay': 0.0,
    'use_amp': True,
    'amp_dtype': 'float16',
    'use_reduce_lr_on_plateau': False,
    'lr_plateau_factor': 0.5,
    'lr_plateau_patience': 10,
    'lr_plateau_threshold': 1e-4,
    'lr_plateau_min_lr': 1e-6,
    'early_stopping_patience': 10,
    'early_stopping_min_delta': 0.0,
    'early_stopping_restore_best': True,
    'transformer_heads': 8,
    'transformer_ff_size': 2048,
    'transformer_dropout': 0.1,
    'train_ratio': 0.75,
    'save_every': 10,
    'diagnostics_every': 1,
    'kl_anneal_enabled': True,
    'kl_anneal_start_beta': 0.0,
    'kl_anneal_max_beta': 1.0,
    'kl_anneal_hold_epochs': 0,
    'kl_anneal_warmup_epochs': 50,
}


# Build standardization tools once (best effort).
_UNCHARGER = None
_TAUTOMER_ENUM = None
if rdMolStandardize is not None:
    try:
        _UNCHARGER = rdMolStandardize.Uncharger()
    except Exception:
        _UNCHARGER = None
    try:
        _TAUTOMER_ENUM = rdMolStandardize.TautomerEnumerator()
    except Exception:
        _TAUTOMER_ENUM = None


def _largest_fragment(mol:Chem.Mol) -> tuple[Chem.Mol, bool]:
    """Keep the largest fragment (salt stripping) and report if stripping happened."""
    try:
        frags = Chem.GetMolFrags(mol, asMols=True)
    except Exception:
        return mol, False

    if not frags or len(frags) <= 1:
        return mol, False

    best = max(frags, key=lambda m: int(m.GetNumHeavyAtoms()))
    return best, True


def canonicalize_for_filtering(
    smiles:str,
    *,
    strip_salts: bool = True,
    decharge: bool = True,
    canonicalize_tautomer: bool = False,
) -> tuple[Optional[str], Optional[Chem.Mol], dict]:
    """Return canonicalized parent SMILES + mol for robust filtering.

    Pipeline (best effort):
      1) Parse/decode from SMILES
      2) Remove salts/mixtures by selecting largest fragment
      3) Neutralize where possible
      4) Optionally canonicalize tautomer representation
      5) Return canonical SMILES used for duplicate/novelty checks
    """
    info = {
        'salt_stripped': 0,
        'tautomer_canonicalized': 0,
    }

    if smiles is None:
        return None, None, info

    s = str(smiles).strip()
    if not s:
        return None, None, info

    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None, None, info

    if strip_salts:
        mol, stripped = _largest_fragment(mol)
        if stripped:
            info['salt_stripped'] = 1

    if decharge and _UNCHARGER is not None:
        try:
            mol = _UNCHARGER.uncharge(mol)
        except Exception:
            pass

    if canonicalize_tautomer and _TAUTOMER_ENUM is not None:
        try:
            before = Chem.MolToSmiles(mol, canonical=True)
            mol = _TAUTOMER_ENUM.Canonicalize(mol)
            after = Chem.MolToSmiles(mol, canonical=True)
            if before != after:
                info['tautomer_canonicalized'] = 1
        except Exception:
            pass

    can = Chem.MolToSmiles(mol, canonical=True)
    if not can:
        return None, None, info
    return can, mol, info


def _flatten_grouped_train_config(config_override:dict) -> dict:
    """Map grouped config sections to legacy flat keys.

    Supported top-level grouped sections:
    - data
    - model
    - transformer
    - optimization
    - training
    - scheduler
    - kl
    - diagnostics
    """
    flat = {}

    data = config_override.get('data')
    if isinstance(data, dict):
        if 'prop_file' in data:
            flat['prop_file'] = data['prop_file']
        if 'seq_length' in data:
            flat['seq_length'] = data['seq_length']
        if 'train_ratio' in data:
            flat['train_ratio'] = data['train_ratio']

    model = config_override.get('model')
    if isinstance(model, dict):
        if 'mode' in model:
            flat['model_mode'] = model['mode']
        if 'latent_size' in model:
            flat['latent_size'] = model['latent_size']
        if 'unit_size' in model:
            flat['unit_size'] = model['unit_size']
        if 'n_rnn_layer' in model:
            flat['n_rnn_layer'] = model['n_rnn_layer']
        if 'num_prop' in model:
            flat['num_prop'] = model['num_prop']
        if 'mean' in model:
            flat['mean'] = model['mean']
        if 'stddev' in model:
            flat['stddev'] = model['stddev']

    transformer = config_override.get('transformer')
    if isinstance(transformer, dict):
        if 'heads' in transformer:
            flat['transformer_heads'] = transformer['heads']
        if 'ff_size' in transformer:
            flat['transformer_ff_size'] = transformer['ff_size']
        if 'dropout' in transformer:
            flat['transformer_dropout'] = transformer['dropout']

    optimization = config_override.get('optimization')
    if isinstance(optimization, dict):
        if 'optimizer' in optimization:
            flat['optimizer'] = optimization['optimizer']
        if 'lr' in optimization:
            flat['lr'] = optimization['lr']
        if 'weight_decay' in optimization:
            flat['weight_decay'] = optimization['weight_decay']
        if 'grad_clip_norm' in optimization:
            flat['grad_clip_norm'] = optimization['grad_clip_norm']
        if 'use_amp' in optimization:
            flat['use_amp'] = optimization['use_amp']
        if 'amp_dtype' in optimization:
            flat['amp_dtype'] = optimization['amp_dtype']

    training = config_override.get('training')
    if isinstance(training, dict):
        if 'batch_size' in training:
            flat['batch_size'] = training['batch_size']
        if 'num_epochs' in training:
            flat['num_epochs'] = training['num_epochs']
        if 'save_dir' in training:
            flat['save_dir'] = training['save_dir']
        if 'run_name' in training:
            flat['run_name'] = training['run_name']
        if 'use_run_subdir' in training:
            flat['use_run_subdir'] = training['use_run_subdir']
        if 'save_every' in training:
            flat['save_every'] = training['save_every']
        if 'early_stopping_patience' in training:
            flat['early_stopping_patience'] = training['early_stopping_patience']
        if 'early_stopping_min_delta' in training:
            flat['early_stopping_min_delta'] = training['early_stopping_min_delta']
        if 'early_stopping_restore_best' in training:
            flat['early_stopping_restore_best'] = training['early_stopping_restore_best']

    scheduler = config_override.get('scheduler')
    if isinstance(scheduler, dict):
        if 'enabled' in scheduler:
            flat['use_reduce_lr_on_plateau'] = scheduler['enabled']
        if 'factor' in scheduler:
            flat['lr_plateau_factor'] = scheduler['factor']
        if 'patience' in scheduler:
            flat['lr_plateau_patience'] = scheduler['patience']
        if 'threshold' in scheduler:
            flat['lr_plateau_threshold'] = scheduler['threshold']
        if 'min_lr' in scheduler:
            flat['lr_plateau_min_lr'] = scheduler['min_lr']

    kl = config_override.get('kl')
    if isinstance(kl, dict):
        if 'enabled' in kl:
            flat['kl_anneal_enabled'] = kl['enabled']
        if 'start_beta' in kl:
            flat['kl_anneal_start_beta'] = kl['start_beta']
        if 'max_beta' in kl:
            flat['kl_anneal_max_beta'] = kl['max_beta']
        if 'hold_epochs' in kl:
            flat['kl_anneal_hold_epochs'] = kl['hold_epochs']
        if 'warmup_epochs' in kl:
            flat['kl_anneal_warmup_epochs'] = kl['warmup_epochs']

    diagnostics = config_override.get('diagnostics')
    if isinstance(diagnostics, dict):
        if 'every' in diagnostics:
            flat['diagnostics_every'] = diagnostics['every']

    return flat


def _normalize_train_config(config_override:dict) -> dict:
    config = get_train_config_defaults()
    grouped = _flatten_grouped_train_config(config_override)
    config.update(grouped)

    # Flat keys still override grouped sections.
    for key in config.keys():
        if key in config_override:
            config[key] = config_override[key]

    # Legacy typo kept for backward compatibility.
    if 'patience' in config_override and 'patientce' not in config_override:
        config['patientce'] = config_override['patience']

    config['model_mode'] = str(config.get('model_mode', 'lstm')).lower()
    if config['model_mode'] not in ('lstm', 'transformer'):
        raise ValueError("model_mode must be either 'lstm' or 'transformer'")
    config['optimizer'] = str(config.get('optimizer', 'adam')).lower()
    if config['optimizer'] not in ('adam', 'adamw'):
        raise ValueError("optimizer must be either 'adam' or 'adamw'")
    config['amp_dtype'] = str(config.get('amp_dtype', 'float16')).lower()
    if config['amp_dtype'] not in ('float16', 'bfloat16'):
        raise ValueError("amp_dtype must be either 'float16' or 'bfloat16'")

    config['training_preset'] = str(config.get('training_preset', 'custom')).strip().lower()
    config['batch_size'] = int(config['batch_size'])
    config['latent_size'] = int(config['latent_size'])
    config['unit_size'] = int(config['unit_size'])
    config['n_rnn_layer'] = int(config['n_rnn_layer'])
    config['seq_length'] = int(config['seq_length'])
    config['prop_file'] = str(config['prop_file'])
    config['mean'] = float(config['mean'])
    config['stddev'] = float(config['stddev'])
    config['num_epochs'] = int(config['num_epochs'])
    config['lr'] = float(config['lr'])
    config['grad_clip_norm'] = float(config.get('grad_clip_norm', 1.0))
    if config.get('num_prop') is None:
        config['num_prop'] = None
    else:
        config['num_prop'] = int(config['num_prop'])
    config['save_dir'] = str(config['save_dir'])
    raw_run_name = config.get('run_name', None)
    if raw_run_name is None:
        config['run_name'] = None
    else:
        run_name = str(raw_run_name).strip()
        config['run_name'] = run_name if run_name else None
    config['use_run_subdir'] = bool(config.get('use_run_subdir', True))
    config['save_every'] = int(config.get('save_every', 10))
    config['patientce'] = int(config.get('patientce', config.get('early_stopping_patience', 10)))
    config['weight_decay'] = float(config.get('weight_decay', 0.0))
    config['use_amp'] = bool(config.get('use_amp', True))
    config['use_reduce_lr_on_plateau'] = bool(config.get('use_reduce_lr_on_plateau', False))
    config['lr_plateau_factor'] = float(config.get('lr_plateau_factor', 0.5))
    config['lr_plateau_patience'] = int(config.get('lr_plateau_patience', 10))
    config['lr_plateau_threshold'] = float(config.get('lr_plateau_threshold', 1e-4))
    config['lr_plateau_min_lr'] = float(config.get('lr_plateau_min_lr', 1e-6))
    config['early_stopping_patience'] = int(config.get('early_stopping_patience', config['patientce']))
    config['early_stopping_min_delta'] = float(config.get('early_stopping_min_delta', 0.0))
    config['early_stopping_restore_best'] = bool(config.get('early_stopping_restore_best', True))
    config['transformer_heads'] = int(config['transformer_heads'])
    config['transformer_ff_size'] = int(config['transformer_ff_size'])
    config['transformer_dropout'] = float(config['transformer_dropout'])
    config['train_ratio'] = float(config.get('train_ratio', 0.75))
    config['diagnostics_every'] = int(config.get('diagnostics_every', 1))
    config['kl_anneal_enabled'] = bool(config.get('kl_anneal_enabled', True))
    config['kl_anneal_start_beta'] = float(config.get('kl_anneal_start_beta', 0.0))
    config['kl_anneal_max_beta'] = float(config.get('kl_anneal_max_beta', 1.0))
    config['kl_anneal_hold_epochs'] = int(config.get('kl_anneal_hold_epochs', 0))
    config['kl_anneal_warmup_epochs'] = int(config.get('kl_anneal_warmup_epochs', 50))

    return config


def sanitize_run_name(name:str) -> str:
    """Convert a run name into a filesystem-friendly folder name."""
    value = str(name).strip()
    if not value:
        raise ValueError('run_name cannot be empty after stripping whitespace.')

    value = re.sub(r'[<>:"/\\|?*]+', '_', value)
    value = re.sub(r'\s+', '_', value)
    value = value.strip('._')
    if not value:
        raise ValueError('run_name became empty after sanitization.')
    return value


def build_train_run_save_dir(
    base_save_dir:str,
    *,
    run_name:Optional[str] = None,
    use_run_subdir:bool = True,
) -> str:
    """Return the effective save directory for a training run."""
    base = str(base_save_dir)
    if not bool(use_run_subdir):
        return base

    if run_name is None:
        auto_name = time.strftime('run_%Y%m%d_%H%M%S')
    else:
        auto_name = sanitize_run_name(str(run_name))

    return os.path.join(base, auto_name)


def resolve_checkpoint_path(
    *,
    save_file:Optional[str] = None,
    run_dir:Optional[str] = None,
    checkpoint_glob:str = 'model_best.ckpt-*.pt',
) -> str:
    """Resolve checkpoint path from explicit file or a run directory."""
    if save_file is not None:
        resolved = str(save_file)
        if not os.path.exists(resolved):
            raise FileNotFoundError(f'Checkpoint not found: {resolved}')
        return resolved

    if run_dir is None:
        raise ValueError('Either save_file or run_dir must be provided.')

    run_dir = str(run_dir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f'Run directory not found: {run_dir}')

    primary_pattern = os.path.join(run_dir, str(checkpoint_glob))
    primary_matches = [p for p in glob.glob(primary_pattern) if os.path.isfile(p)]
    if primary_matches:
        primary_matches.sort(key=os.path.getmtime, reverse=True)
        return primary_matches[0]

    fallback_pattern = os.path.join(run_dir, '*.pt')
    fallback_matches = [p for p in glob.glob(fallback_pattern) if os.path.isfile(p)]
    if fallback_matches:
        fallback_matches.sort(key=os.path.getmtime, reverse=True)
        return fallback_matches[0]

    raise FileNotFoundError(
        f'No checkpoint files found in run_dir={run_dir} (pattern={checkpoint_glob!r}).'
    )


def load_training_canonical_smiles(
    prop_file:str,
    seq_length:int,
    *,
    strip_salts: bool = True,
    decharge: bool = True,
    canonicalize_tautomer: bool = False,
) -> set:
    """Load canonical SMILES set from the training/property file.

    This follows the same length filter used in `load_data` so the exclusion set
    matches what the model was trained on.

    The goal of this is to cache the canonicalized SMILES
    """
    # Cache to avoid re-canonicalizing large training files on every run.
    # Versioned cache suffix so stale canonicalization from older logic isn't reused.
    cache_path = (
        f"{prop_file}.canon_std_v3"
        f"_salts{int(bool(strip_salts))}"
        f"_dechg{int(bool(decharge))}"
        f"_taut{int(bool(canonicalize_tautomer))}"
        f"_seq{int(seq_length)}.pkl.gz"
    )
    try:
        if os.path.exists(cache_path):
            cache_mtime = os.path.getmtime(cache_path)
            src_mtime = os.path.getmtime(prop_file)
            if cache_mtime >= src_mtime:
                cached = load_pickle_gz(cache_path)
                if isinstance(cached, set):
                    return cached
    except Exception:
        # Cache is best-effort; fall back to recompute.
        pass

    canonical = set()
    with open(prop_file) as f:
        lines = f.read().split('\n')[:-1]

    lines = [l.split() for l in lines]
    lines = [l for l in lines if len(l) > 0 and len(l[0]) < seq_length - 2]

    for row in lines:
        smi = row[0]
        can, _, _ = canonicalize_for_filtering(
            smi,
            strip_salts=strip_salts,
            decharge=decharge,
            canonicalize_tautomer=canonicalize_tautomer,
        )
        if can is not None:
            canonical.add(can)

    try:
        save_pickle_gz(cache_path, canonical)
    except Exception:
        pass
    return canonical


def collect_new_unique_from_raw(
    raw_strings:list,
    seen_smiles:set,
    training_smiles:Optional[set] = None,
    eos_token:str = 'E',
    accept_predicate: Optional[Callable[[str, Chem.Mol], bool]] = None,
    strip_salts: bool = True,
    decharge: bool = True,
    canonicalize_tautomer: bool = False,
) -> tuple:
    """Filter decoded strings into new unique molecules + quality stats.

    Returns:
      - accepted: list of (canonical_smiles, mol)
      - stats: dict with counters for quality reporting
    """
    accepted = []
    stats = {
        'total_generated': 0,
        'accepted': 0,
        'invalid_or_empty': 0,
        'in_training': 0,
        'duplicate': 0,
        'rejected_by_filter': 0,
        'salt_stripped': 0,
        'tautomer_canonicalized': 0,
    }

    if training_smiles is None:
        training_smiles = set()

    for s in raw_strings:
        stats['total_generated'] += 1

        # Drop everything after EOS/padding.
        s = s.split(eos_token)[0].strip()
        if not s:
            stats['invalid_or_empty'] += 1
            continue

        # Robust decode + normalization before novelty/duplicate checks.
        # This strips salts/fragments and canonicalizes tautomer forms so
        # duplicates and novelty are measured on a consistent parent form.
        can, mol, can_info = canonicalize_for_filtering(
            s,
            strip_salts=strip_salts,
            decharge=decharge,
            canonicalize_tautomer=canonicalize_tautomer,
        )
        if can is None or mol is None:
            stats['invalid_or_empty'] += 1
            continue

        stats['salt_stripped'] += int(can_info.get('salt_stripped', 0))
        stats['tautomer_canonicalized'] += int(can_info.get('tautomer_canonicalized', 0))

        if can in training_smiles:
            stats['in_training'] += 1
            continue

        if can in seen_smiles:
            stats['duplicate'] += 1
            continue

        if accept_predicate is not None:
            try:
                ok = bool(accept_predicate(can, mol))
            except Exception:
                ok = False
            if not ok:
                stats['rejected_by_filter'] += 1
                continue

        seen_smiles.add(can)
        accepted.append((can, mol))
        stats['accepted'] += 1

    return accepted, stats

def convert_to_smiles(vector:np.ndarray, char:np.ndarray) -> str:
    """
    Converts a vector of one-hot encodings back to a SMILES string using the provided character set.
    Args:
        vector (np.ndarray): A 2D array of shape (sequence_length, vocab_size) representing one-hot encodings.
        char (np.ndarray): A 1D array of shape (vocab_size,) containing the characters corresponding to the one-hot encodings.
    
    """
    list_char = list(char)
    #list_char = char.tolist()
    vector = vector.astype(int)
    return "".join(map(lambda x: list_char[x], vector)).strip()


def load_sampling_metadata(prop_file: str, seq_length: int) -> tuple[np.ndarray, dict, int]:
    """Load charset/vocab/num_prop for sampling without building training tensors.

    Uses a gzip-pickle cache keyed by file + seq length to make repeated startup
    for large property files fast.
    """
    cache_path = f"{prop_file}.sample_meta_seq{int(seq_length)}.pkl.gz"
    try:
        if os.path.exists(cache_path):
            cache_mtime = os.path.getmtime(cache_path)
            src_mtime = os.path.getmtime(prop_file)
            if cache_mtime >= src_mtime:
                cached = load_pickle_gz(cache_path)
                chars_cached = cached.get('charset')
                vocab_cached = cached.get('vocab')
                num_prop_cached = cached.get('num_prop')
                if chars_cached is not None and vocab_cached is not None and num_prop_cached is not None:
                    return np.array(chars_cached), dict(vocab_cached), int(num_prop_cached)
    except Exception:
        pass

    import collections

    counter = collections.Counter()
    num_prop: Optional[int] = None

    with open(prop_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            tokens = line.split()
            if not tokens:
                continue
            smi = tokens[0]
            if len(smi) >= int(seq_length) - 2:
                continue

            if num_prop is None:
                num_prop = max(0, len(tokens) - 1)
            counter.update(smi)

    if num_prop is None:
        raise ValueError(f'No usable rows found in prop file: {prop_file}')

    if len(counter) == 0:
        raise ValueError(
            f'No SMILES passed length filter for seq_length={int(seq_length)} in prop file: {prop_file}'
        )

    chars = tuple([ch for ch, _ in counter.most_common()])
    vocab = dict(zip(chars, range(len(chars))))
    chars += ('E',)
    chars += ('X',)
    vocab['E'] = len(chars) - 2
    vocab['X'] = len(chars) - 1

    payload = {
        'charset': chars,
        'vocab': vocab,
        'num_prop': int(num_prop),
    }
    try:
        save_pickle_gz(cache_path, payload)
    except Exception:
        pass

    return np.array(chars), vocab, int(num_prop)

def load_data(n:str, seq_length:int) -> tuple:
    import collections
    with open(n, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.read().split('\n')[:-1]
    lines = [l.split() for l in lines]
    lines = [l for l in lines if len(l[0])<seq_length-2]
    smiles = [l[0] for l in lines]
    
    total_string = ''
    for s in smiles:
        total_string+=s
    counter = collections.Counter(total_string)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    chars, counts = zip(*count_pairs)
    vocab = dict(zip(chars, range(len(chars))))

    chars+=('E',) #End of smiles
    chars+=('X',) #Start of smiles
    vocab['E'] = len(chars)-2
    vocab['X'] = len(chars)-1
    
    length = np.array([len(s)+1 for s in smiles])
    smiles_input = [('X'+s).ljust(seq_length, 'E') for s in smiles] 
    smiles_output = [s.ljust(seq_length, 'E') for s in smiles] 
    smiles_input = np.array([np.array(list(map(vocab.get, s)))for s in smiles_input])
    smiles_output = np.array([np.array(list(map(vocab.get, s)))for s in smiles_output])
    prop = np.array([l[1:] for l in lines], dtype=np.float32)
    return smiles_input, smiles_output, chars, vocab, prop, length 


def get_train_config_defaults() -> dict:
    return dict(TRAIN_CONFIG_DEFAULTS)


def compose_train_config(args) -> dict:
    """Compose training config from defaults + optional JSON + CLI overrides."""
    raw = {}

    config_file = getattr(args, 'config_file', None)
    if config_file:
        loaded = load_json(config_file)
        if isinstance(loaded, dict):
            raw.update(loaded)

    for key in get_train_config_defaults().keys():
        if hasattr(args, key):
            value = getattr(args, key)
            if value is not None:
                raw[key] = value

    return _normalize_train_config(raw)


def compose_train_config_from_dict(config_override:dict) -> dict:
    """Compose training config from defaults + in-file dict overrides."""
    if config_override is None:
        config_override = {}
    return _normalize_train_config(config_override)


def get_model_config(config:dict, vocab_size:Optional[int] = None) -> dict:
    """Keep essential config values needed to recreate model architecture.

    Note: In transformer mode, token embedding size is derived from latent_size.
    """
    model_config = {
        'batch_size': int(config['batch_size']),
        'latent_size': int(config['latent_size']),
        'unit_size': int(config['unit_size']),
        'n_rnn_layer': int(config['n_rnn_layer']),
        'seq_length': int(config['seq_length']),
        'mean': float(config['mean']),
        'stddev': float(config['stddev']),
        'lr': float(config['lr']),
        'num_prop': int(config['num_prop']),
        'grad_clip_norm': float(config.get('grad_clip_norm', 1.0)),
        'model_mode': str(config.get('model_mode', 'lstm')).lower(),
        'optimizer': str(config.get('optimizer', 'adam')).lower(),
        'weight_decay': float(config.get('weight_decay', 0.0)),
        'use_amp': bool(config.get('use_amp', True)),
        'amp_dtype': str(config.get('amp_dtype', 'float16')).lower(),
        'transformer_heads': int(config.get('transformer_heads', 8)),
        'transformer_ff_size': int(config.get('transformer_ff_size', int(config['unit_size']) * 4)),
        'transformer_dropout': float(config.get('transformer_dropout', 0.1)),
        'prop_file': str(config.get('prop_file', 'smiles_prop.txt')),
        'save_dir': str(config.get('save_dir', 'save/')),
        'run_name': config.get('run_name', None),
        'use_run_subdir': bool(config.get('use_run_subdir', True)),
    }
    if vocab_size is not None:
        model_config['vocab_size'] = int(vocab_size)
    return model_config


def save_json(path:str, payload:dict) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def save_pickle_gz(
    path: str,
    payload: Any,
    *,
    protocol: int = pickle.HIGHEST_PROTOCOL,
    compresslevel: int = 6,
) -> str:
    """Serialize payload to a gzip-compressed pickle file.

    This is intended for large Python objects (e.g., generated molecule lists)
    where plain text/CSV outputs can become very large.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with gzip.open(path, 'wb', compresslevel=int(compresslevel)) as f:
        pickle.dump(payload, f, protocol=protocol)
    return path


def load_pickle_gz(path: str) -> Any:
    """Load and return an object from a gzip-compressed pickle file."""
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)


def load_checkpoint_model_config(ckpt_path: str) -> Optional[dict]:
    """Load `model_config` embedded in a '.pt' checkpoint, if present.

    This is more reliable than relying on 'save/training_config.json', which can
    be overwritten by later training runs and silently mismatch the checkpoint.
    """
    try:
        import torch
        import inspect

        kwargs: dict[str, Any] = {'map_location': 'cpu'}
        if 'weights_only' in inspect.signature(torch.load).parameters:
            # Keep this safe-loader path; it still supports dict/list/str/float payloads.
            kwargs['weights_only'] = True
        checkpoint = torch.load(ckpt_path, **kwargs)
        if isinstance(checkpoint, dict):
            cfg = checkpoint.get('model_config')
            return cfg if isinstance(cfg, dict) else None
    except Exception:
        return None
    return None


def load_json(path:str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_training_config(config:dict, save_dir:str, filename:str = 'training_config.json') -> str:
    path = os.path.join(save_dir, filename)
    save_json(path, config)
    return path


def infer_training_config_path(ckpt_path:str, filename:str = 'training_config.json') -> str:
    return os.path.join(os.path.dirname(ckpt_path), filename)


def ensure_dir(path:str) -> None:
    os.makedirs(path, exist_ok=True)


def split_train_test(data:np.ndarray, train_ratio:float = 0.75) -> tuple:
    num_train = int(len(data) * train_ratio)
    return data[0:num_train], data[num_train:]

