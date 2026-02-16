from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcNumHBD
from rdkit.Chem.rdMolDescriptors import CalcNumHBA
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit import Chem
from multiprocessing import Pool

#parser = argparse.ArgumentParser()
#parser.add_argument('--input_filename', help='filename for smiles', type=str, default='smiles.txt')
#parser.add_argument('--output_filename', help='name of output file', type=str, default='smiles_prop.txt')
#parser.add_argument('--ncpus', help='number of cpus', type=int, default=1)
#args = parser.parse_args()


#do args using a config dict:

args = {
    'input_filename' : 'smiles.txt',
    'output_filename' : 'prop_mw_logp.txt',
    # order matters: this defines the conditioning column order for train/sample.
    "properties": ['MW', 'LogP'], # any subset/order of: MW, LogP, TPSA, NumHBD, NumHBA
    'ncpus' : 1
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


def read_smiles(filename: str) -> list:
    with open(filename) as f:
        smiles = f.read().split('\n')[:-1]
    return smiles


if __name__ == '__main__':
    _validate_properties(args['properties'])
    smiles = read_smiles(args['input_filename'])
    
    pool = Pool(args['ncpus'])
    print('Calculating properties for %d molecules...' % len(smiles))
    r = pool.map_async(cal_prop, smiles)

    data = r.get()
    pool.close()
    pool.join()
    with open(args['output_filename'], 'w') as w:
        for i, d in enumerate(data):
            if d is None:
                continue
            w.write('\t'.join(map(str, d)) + '\n')
            if i % 1000 == 0:
                print('Processed %d molecules...' % i)

