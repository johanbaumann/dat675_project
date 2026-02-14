from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcNumHBD
from rdkit.Chem.rdMolDescriptors import CalcNumHBA
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit import Chem
from multiprocessing import Pool
import argparse

#parser = argparse.ArgumentParser()
#parser.add_argument('--input_filename', help='filename for smiles', type=str, default='smiles.txt')
#parser.add_argument('--output_filename', help='name of output file', type=str, default='smiles_prop.txt')
#parser.add_argument('--ncpus', help='number of cpus', type=int, default=1)
#args = parser.parse_args()


#do args using a config dict:

args = {
    'input_filename' : 'smiles.txt',
    'output_filename' : 'smiles_prop.txt',
    'ncpus' : 1
}




def cal_prop(s: str) -> tuple:
    m = Chem.MolFromSmiles(s)
    if m is None : return None
    return Chem.MolToSmiles(m), ExactMolWt(m), MolLogP(m), CalcTPSA(m)
    #return Chem.MolToSmiles(m), ExactMolWt(m), MolLogP(m), CalcNumHBD(m), CalcNumHBA(m), CalcTPSA(m)

def read_smiles(filename: str) -> list:
    with open(filename) as f:
        smiles = f.read().split('\n')[:-1]
    return smiles

if __name__ == '__main__':
    smiles = read_smiles(args['input_filename'])
    
    pool = Pool(args['ncpus'])
    print('Calculating properties for %d molecules...' % len(smiles))
    r = pool.map_async(cal_prop, smiles)

    data = r.get()
    pool.close()
    pool.join()
    w = open(args['output_filename'], 'w')

    for i, d in enumerate(data):
        if d is None:
            continue
        w.write(d[0] + '\t' + str(d[1]) + '\t'+ str(d[2]) + '\t'+ str(d[3]) + '\n')
        if i % 1000 == 0:
            print('Processed %d molecules...' % i)
    w.close()

