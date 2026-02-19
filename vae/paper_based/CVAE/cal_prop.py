from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcNumHBD
from rdkit.Chem.rdMolDescriptors import CalcNumHBA
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit import Chem
from multiprocessing import Pool
import pandas as pd

#parser = argparse.ArgumentParser()
#parser.add_argument('--input_filename', help='filename for smiles', type=str, default='smiles.txt')
#parser.add_argument('--output_filename', help='name of output file', type=str, default='smiles_prop.txt')
#parser.add_argument('--ncpus', help='number of cpus', type=int, default=1)
#args = parser.parse_args()


#do args using a config dict:

args = {
    'input_filename' : '250k_rndm_zinc_drugs_clean_3.csv',
    'output_filename' : '250k_zinc_clean.txt',
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
    # if the user specified no properties, all will be calculated...
    if len(selected) == 0:
        raise ValueError('args["properties"] must contain at least one descriptor name.')
    unknown = [p for p in selected if p not in PROPERTY_FUNCTIONS]
    if unknown:
        supported = ', '.join(PROPERTY_FUNCTIONS.keys())
        raise ValueError(f'Unknown property names: {unknown}. Supported: {supported}')



# is applied to each smiles string in the input file
def cal_prop(s: str) -> tuple:
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    props = [PROPERTY_FUNCTIONS[name](m) for name in args['properties']]
    return Chem.MolToSmiles(m), *props

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
    # get the file type and read accordingly. for now we only support a single column of smiles, no header, and ignore blank lines.
    file_type = get_file_type(filename)
    if file_type == 'txt':
       
    
        with open(filename) as f:
            # we assume the file is a single column of smiles, no header, and ignore blank lines
            smiles = f.read().split('\n')[:-1]
        return smiles
    elif file_type == 'csv':
        import pandas as pd
        df = pd.read_csv(filename)
        # we assume the file has a column named 'smiles' containing the smiles strings
        if 'smiles' not in df.columns:
            raise ValueError('CSV input file must contain a "smiles" column.')
        return df['smiles'].tolist()

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

