#import h5py
import numpy as np
from rdkit import Chem
from typing import Optional, Sequence, Union
import json
import os
import importlib


TRAIN_CONFIG_DEFAULTS = {
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
    'save_dir': 'save/',
    'patientce': 10,
    'model_mode': 'lstm',
    'optimizer': 'adam',
    'weight_decay': 0.0,
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
}


def load_training_canonical_smiles(prop_file:str, seq_length:int) -> set:
    """Load canonical SMILES set from the training/property file.

    This follows the same length filter used in `load_data` so the exclusion set
    matches what the model was trained on.
    """
    canonical = set()
    with open(prop_file) as f:
        lines = f.read().split('\n')[:-1]

    lines = [l.split() for l in lines]
    lines = [l for l in lines if len(l) > 0 and len(l[0]) < seq_length - 2]

    for row in lines:
        smi = row[0]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        can = Chem.MolToSmiles(mol)
        if can:
            canonical.add(can)
    return canonical


def collect_new_unique_from_raw(
    raw_strings:list,
    seen_smiles:set,
    training_smiles:Optional[set] = None,
    eos_token:str = 'E',
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

        mol = Chem.MolFromSmiles(s)
        if mol is None:
            stats['invalid_or_empty'] += 1
            continue

        can = Chem.MolToSmiles(mol)
        if not can:
            stats['invalid_or_empty'] += 1
            continue

        if can in training_smiles:
            stats['in_training'] += 1
            continue

        if can in seen_smiles:
            stats['duplicate'] += 1
            continue

        seen_smiles.add(can)
        accepted.append((can, mol))
        stats['accepted'] += 1

    return accepted, stats

def convert_to_smiles(vector:np.ndarray, char:np.ndarray) -> str:
    list_char = list(char)
    #list_char = char.tolist()
    vector = vector.astype(int)
    return "".join(map(lambda x: list_char[x], vector)).strip()

def stochastic_convert_to_smiles(vector:np.ndarray, char:np.ndarray) -> str:
    list_char = char.tolist()
    s = ""
    for i in range(len(vector)):
        prob = vector[i].tolist()
        norm0 = sum(prob)
        prob = [i/norm0 for i in prob]
        index = np.random.choice(len(list_char), 1, p=prob)
        s+=list_char[index[0]]
    return s

def one_hot_array(i:int, n:int) -> list:
    return list(map(int, [ix == i for ix in range(n)]))

def one_hot_index(vec:Union[np.ndarray, Sequence[str]], charset:str) -> list:
    return list(map(charset.index, vec))

def from_one_hot_array(vec:np.ndarray) -> Optional[int]:
    oh = np.where(vec == 1)
    if oh[0].shape == (0, ):
        return None
    return int(oh[0][0])

def decode_smiles_from_indexes(vec:np.ndarray, charset:str) -> str:
    return "".join(map(lambda x: charset[x], vec)).strip()

def load_dataset(filename:str, split:bool = True) -> tuple:
    h5py = importlib.import_module('h5py')
    h5f = h5py.File(filename, 'r')
    if split:
        data_train = h5f['data_train'][:]
    else:
        data_train = None
    data_test = h5f['data_test'][:]
    charset = h5f['charset'][:]
    h5f.close()
    if split:
        return data_train, data_test, charset
    else:
        return data_test, charset

def encode_smiles(smiles:str, model, charset:str) -> np.ndarray:
    cropped = list(smiles.ljust(120))
    preprocessed = np.array([list(map(lambda x: one_hot_array(x, len(charset)), one_hot_index(cropped, charset)))])
    latent = model.encoder.predict(preprocessed)
    return latent

def smiles_to_onehot(smiles:str, charset:str) -> np.ndarray:
    cropped = list(smiles.ljust(120))
    preprocessed = np.array([list(map(lambda x: one_hot_array(x, len(charset)), one_hot_index(cropped, charset)))])
    return preprocessed

def smiles_to_vector(smiles:str, vocab:dict, max_length:int) -> list:
    while len(smiles)<max_length:
        smiles +=" "
    return [vocab[str(x)] for x in smiles]

def decode_latent_molecule(latent:np.ndarray, model, charset:str, latent_dim:int) -> str:
    decoded = model.decoder.predict(latent.reshape(1, latent_dim)).argmax(axis=2)[0]
    smiles = decode_smiles_from_indexes(decoded, charset)
    return smiles

def interpolate(source_smiles:str, dest_smiles:str, steps:int, charset:str, model, latent_dim:int) -> list:
    source_latent = encode_smiles(source_smiles, model, charset)
    dest_latent = encode_smiles(dest_smiles, model, charset)
    step = (dest_latent - source_latent) / float(steps)
    results = []
    for i in range(steps):
        item = source_latent + (step * i)        
        decoded = decode_latent_molecule(item, model, charset, latent_dim)
        results.append(decoded)
    return results

def get_unique_mols(mol_list:list) -> list:
    inchi_keys = []
    for mol in mol_list:
        inchi = Chem.MolToInchi(mol)
        if isinstance(inchi, tuple):
            inchi = inchi[0]
        inchi_keys.append(Chem.InchiToInchiKey(str(inchi)))
    u, indices = np.unique(inchi_keys, return_index=True)
    unique_mols = [[mol_list[i], inchi_keys[i]] for i in indices]
    return unique_mols

def accuracy(arr1:np.ndarray, arr2:np.ndarray, length:np.ndarray) -> tuple:
    total = len(arr1)
    count1=0
    count2=0
    count3=0
    for i in range(len(arr1)):
        if np.array_equal(arr1[i,:length[i]], arr2[i,:length[i]]):
            count1+=1
    for i in range(len(arr1)):
        for j in range(length[i]):
            if arr1[i][j]==arr2[i][j]:
                count2+=1
            count3+=1

    return float(count1/float(total)), float(count2/count3)




def load_data(n:str, seq_length:int) -> tuple:
    import collections
    f = open(n)
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
    config = get_train_config_defaults()

    config_file = getattr(args, 'config_file', None)
    if config_file:
        loaded = load_json(config_file)
        config.update(loaded)

    for key in config.keys():
        if not hasattr(args, key):
            continue
        value = getattr(args, key)
        if value is not None:
            config[key] = value

    config['model_mode'] = str(config.get('model_mode', 'lstm')).lower()
    if config['model_mode'] not in ('lstm', 'transformer'):
        raise ValueError("model_mode must be either 'lstm' or 'transformer'")
    config['optimizer'] = str(config.get('optimizer', 'adam')).lower()
    if config['optimizer'] not in ('adam', 'adamw'):
        raise ValueError("optimizer must be either 'adam' or 'adamw'")

    # normalize scalar types
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
    config['num_prop'] = int(config['num_prop'])
    config['save_dir'] = str(config['save_dir'])
    config['patientce'] = int(config['patientce'])
    config['weight_decay'] = float(config.get('weight_decay', 0.0))
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
    return config


def compose_train_config_from_dict(config_override:dict) -> dict:
    """Compose training config from defaults + in-file dict overrides."""
    config = get_train_config_defaults()
    config.update(config_override)

    config['model_mode'] = str(config.get('model_mode', 'lstm')).lower()
    if config['model_mode'] not in ('lstm', 'transformer'):
        raise ValueError("model_mode must be either 'lstm' or 'transformer'")
    config['optimizer'] = str(config.get('optimizer', 'adam')).lower()
    if config['optimizer'] not in ('adam', 'adamw'):
        raise ValueError("optimizer must be either 'adam' or 'adamw'")

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
    config['num_prop'] = int(config['num_prop'])
    config['save_dir'] = str(config['save_dir'])
    config['patientce'] = int(config['patientce'])
    config['weight_decay'] = float(config.get('weight_decay', 0.0))
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
    return config


def build_train_config(args) -> dict:
    """Backward-compatible alias for compose_train_config."""
    return compose_train_config(args)


def get_model_config(config:dict, vocab_size:Optional[int] = None) -> dict:
    """Keep essential config values needed to recreate model architecture."""
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
        'model_mode': str(config.get('model_mode', 'lstm')).lower(),
        'optimizer': str(config.get('optimizer', 'adam')).lower(),
        'weight_decay': float(config.get('weight_decay', 0.0)),
        'transformer_heads': int(config.get('transformer_heads', 8)),
        'transformer_ff_size': int(config.get('transformer_ff_size', int(config['unit_size']) * 4)),
        'transformer_dropout': float(config.get('transformer_dropout', 0.1)),
        'prop_file': str(config.get('prop_file', 'smiles_prop.txt')),
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
    return data[0:num_train], data[num_train:-1]

