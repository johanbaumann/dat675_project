from model import CVAE
from utils import *
import numpy as np
import argparse
import os
import time as t
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcTPSA
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', help='batch_size', type=int, default=128)
parser.add_argument('--num_iteration', help='num_iteration', type=int, default=10)
parser.add_argument('--latent_size', help='latent_size', type=int, default=200)
parser.add_argument('--unit_size', help='unit_size of rnn cell', type=int, default=512)
parser.add_argument('--n_rnn_layer', help='number of rnn layer', type=int, default=3)
parser.add_argument('--seq_length', help='max_seq_length', type=int, default=120)
parser.add_argument('--mean', help='mean of VAE', type=float, default=0.0)
parser.add_argument('--stddev', help='stddev of VAE', type=float, default=1.0)
parser.add_argument('--num_prop', help='number of propertoes', type=int, default=3)
parser.add_argument('--save_file', help='save file', type=str)
parser.add_argument('--target_prop', help='target properties', type=str)
parser.add_argument('--prop_file', help='name of property file', type=str)
parser.add_argument('--result_filename', help='name of result filename', type=str, default='result.txt')
parser.add_argument('--lr', help='learning rate', type=float, default=0.0001)
args = parser.parse_args()

start = t.time()
config = {
    'batch_size': 128,
    'num_iteration': 10, # number of iteration to generate smiles
    'latent_size': 200,
    'unit_size': 512,
    'n_rnn_layer': 3,
    'seq_length': 120,
    'mean': 0.0,
    'stddev': 1.0,
    'num_prop': 3,
    'save_file': 'save/model_.ckpt-99.pt',
    'target_prop': '300.0 3.0 75.0', # target properties (MW, LogP, TPSA) to generate smiles. You can change this value to generate different smiles.
    'prop_file': "smiles_prop.txt",
    'result_filename': 'test_laptop_res.txt',
    'lr': 0.0001
}

for key in config:
    value = getattr(args, key, None)
    if value is not None:
        config[key] = value



#convert smiles to numpy array
_, _, char, vocab, _, _ = load_data(config['prop_file'], config['seq_length'])
vocab_size = len(char)

#model and restore model parapmeters
model = CVAE(vocab_size,
             config,
             )
model.restore(config['save_file'])

print('Number of parameters : ', sum(p.numel() for p in model.parameters() if p.requires_grad))

#target property to numpy array
target_prop = np.array([[float(p) for p in config['target_prop'].split()] for _ in range(config['batch_size'])])
start_codon = np.array([np.array(list(map(vocab.get, 'X')))for _ in range(config['batch_size'])])

#generate smiles
smiles = []
for _ in range(config['num_iteration']):
    latent_vector = np.random.normal(config['mean'], config['stddev'], (config['batch_size'], config['latent_size']))
    generated = model.sample(latent_vector, target_prop, start_codon, config['seq_length'])
    smiles += [convert_to_smiles(generated[i], char) for i in range(len(generated))]

#write smiles and calcualte properties of them    
print ('number of trial : ', len(smiles))
smiles = list(set([s.split('E')[0] for s in smiles]    ))
print ('number of generate smiles (after remove duplicated ones) : ', len(smiles))
ms = [Chem.MolFromSmiles(s) for s in smiles]
ms = [m for m in ms if m is not None]
print ('number of valid smiles : ', len(ms))


def avg_mv(mols:list) -> float:
    return sum([ExactMolWt(m) for m in mols])/len(mols)

def avg_logp(mols:list) -> float:
    return sum([MolLogP(m) for m in mols])/len(mols)


smiles = [Chem.MolToSmiles(m) for m in ms]
mw = [ExactMolWt(m) for m in ms]
logp = [MolLogP(m) for m in ms]
tpsa = [CalcTPSA(m) for m in ms]

df = pd.DataFrame({
    'smiles': smiles,
    'MW': mw,
    'LogP': logp,
    'TPSA': tpsa
})

print(df.describe())

print('average MW : ', avg_mv(ms))
print('average LogP : ', avg_logp(ms))






# save smiles and properties to file
df.to_csv(config['result_filename'], index=False)

end_time = t.time()
print(f'time to run: {end_time - start}')


#
#
#with open(config['result_filename'], 'w') as w:
#    w.write('smiles\tMW\tLogP\tTPSA\n')
#    for m in ms:
#        try:
#            w.write('%s\t%.3f\t%.3f\t%.3f\n' %(Chem.MolToSmiles(m), ExactMolWt(m), MolLogP(m), CalcTPSA(m)))
#        except:
#            continue            
