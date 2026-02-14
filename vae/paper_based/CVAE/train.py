from model import CVAE
from utils import *
import numpy as np
import os
import torch
import time
import argparse
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', help='batch_size', type=int, default=128)
parser.add_argument('--latent_size', help='latent_size', type=int, default=200)
parser.add_argument('--unit_size', help='unit_size of rnn cell', type=int, default=512)
parser.add_argument('--n_rnn_layer', help='number of rnn layer', type=int, default=3)
parser.add_argument('--seq_length', help='max_seq_length', type=int, default=120)
parser.add_argument('--prop_file', help='name of property file', type=str)
parser.add_argument('--mean', help='mean of VAE', type=float, default=0.0)
parser.add_argument('--stddev', help='stddev of VAE', type=float, default=1.0)
parser.add_argument('--num_epochs', help='epochs', type=int, default=100)
parser.add_argument('--lr', help='learning rate', type=float, default=0.0001)
parser.add_argument('--num_prop', help='number of propertoes', type=int, default=3)
parser.add_argument('--save_dir', help='save dir', type=str, default='save/')
args = parser.parse_args()




config = {
    'batch_size': 128,
    'latent_size': 200,
    'unit_size': 512,
    'n_rnn_layer': 3,
    'seq_length': 120,
    'prop_file': "smiles_prop.txt",
    'mean': 0.0,
    'stddev': 1.0,
    'num_epochs': 100,
    'lr': 0.0001,
    'num_prop': 3,
    'save_dir': 'save/'
}


print (config)
#convert smiles to numpy array
molecules_input, molecules_output, char, vocab, labels, length = load_data(config['prop_file'], config['seq_length'])
vocab_size = len(char)

#make save_dir
if not os.path.isdir(config['save_dir']):
    os.mkdir(config['save_dir'])

#divide data into training and test set
num_train_data = int(len(molecules_input)*0.75)
train_molecules_input = molecules_input[0:num_train_data]
test_molecules_input = molecules_input[num_train_data:-1]

train_molecules_output = molecules_output[0:num_train_data]
test_molecules_output = molecules_output[num_train_data:-1]

train_labels = labels[0:num_train_data]
test_labels = labels[num_train_data:-1]

train_length = length[0:num_train_data]
test_length = length[num_train_data:-1]

model = CVAE(vocab_size,
             args
             )
print('Number of parameters : ', sum(p.numel() for p in model.parameters() if p.requires_grad))

history = {
    'train_loss': [],
    'test_loss': []
}
    
for epoch in range(args.num_epochs):

    st = time.time()
    # Learning rate scheduling 
    #model.assign_lr(learning_rate * (decay_rate ** epoch))
    train_loss = []
    test_loss = []
    
 
    st = time.time()
    

    #train
    for iteration in range(len(train_molecules_input)//config['batch_size']):
        n = np.random.randint(len(train_molecules_input), size = config['batch_size'])
        x = np.array([train_molecules_input[i] for i in n])
        y = np.array([train_molecules_output[i] for i in n])
        l = np.array([train_length[i] for i in n])
        c = np.array([train_labels[i] for i in n])
        cost = model.train_batch(x, y, l, c)
        train_loss.append(cost)
    # test on test set..
    # there is data leakage here, but we just want to see the trend of loss, not the actual performance of the model.
    for iteration in range(len(test_molecules_input)//config['batch_size']):
        n = np.random.randint(len(test_molecules_input), size = config['batch_size'])
        x = np.array([test_molecules_input[i] for i in n])
        y = np.array([test_molecules_output[i] for i in n])
        l = np.array([test_length[i] for i in n])
        c = np.array([test_labels[i] for i in n])
        cost = model.test_batch(x, y, l, c)
        test_loss.append(cost)
    
    train_loss = np.mean(np.array(train_loss))        
    test_loss = np.mean(np.array(test_loss))    
    history['train_loss'].append(train_loss)
    history['test_loss'].append(test_loss)
    end = time.time()    
    if epoch==0:
        print ('epoch\ttrain_loss\ttest_loss\ttime (s)')
    print ("%s\t%.3f\t%.3f\t%.3f" %(epoch, train_loss, test_loss, end-st))
    # save model!
    # only save for the last epoch...

    if epoch == config['num_epochs']-1:
        ckpt_path = config['save_dir']+'/model_'+'.ckpt'
        model.save(ckpt_path, epoch)
        #save history as csv file
        history_df = pd.DataFrame(history)
        history_df.to_csv(config['save_dir']+'/history.csv', index=False)

