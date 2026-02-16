from model import CVAE
from utils import *
import numpy as np
import time
import pandas as pd


# Single source of truth for run configuration.
# Edit values here directly; CLI arguments are intentionally disabled.
config = {
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
    'model_mode': 'lstm',  # 'lstm' or 'transformer'
    'transformer_heads': 8,
    'transformer_ff_size': 2048,
    'transformer_dropout': 0.1,
    'train_ratio': 0.75,
}

config = compose_train_config_from_dict(config)


print (config)
#convert smiles to numpy array
molecules_input, molecules_output, char, vocab, labels, length = load_data(config['prop_file'], config['seq_length'])
vocab_size = len(char)
model_config = get_model_config(config, vocab_size=vocab_size)

#make save_dir
ensure_dir(config['save_dir'])

# save a single source of truth for recreating the trained model
training_config_path = save_training_config(model_config, config['save_dir'])
print(f'saved training config to: {training_config_path}')

#divide data into training and test set
# can leak...
train_molecules_input, test_molecules_input = split_train_test(molecules_input, config['train_ratio'])
train_molecules_output, test_molecules_output = split_train_test(molecules_output, config['train_ratio'])
train_labels, test_labels = split_train_test(labels, config['train_ratio'])
train_length, test_length = split_train_test(length, config['train_ratio'])

model = CVAE(vocab_size,
             model_config
             )
print('Number of parameters : ', sum(p.numel() for p in model.parameters() if p.requires_grad))

history = {
    'train_loss': [],
    'test_loss': []
}


for epoch in range(config['num_epochs']):

    st = time.time()
    # Learning rate scheduling 
    #model.assign_lr(learning_rate * (decay_rate ** epoch))
    train_loss = []
    test_loss = []
    
 
    st = time.time()
    
    """
    train:
    n: random batch of data
    x: input_smiles 
    y: output_smiles, with added X(start token) and E(end-token) tokens
    l: lenght of input smiles
    c: properties (Conditions) of generation
    
    
    """
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


    # calculate how many epoch since best epoch (lowest test loss)
    epochs_since_best = epoch - np.argmin(history['test_loss'])

    #early stopping if no improvment for a while
    # 
    if epochs_since_best > config['patientce']:
        print(f'early stop at epoch {epoch} since no improvement for {epochs_since_best} epochs')
        
        ckpt_path = config['save_dir']+'/model_'+'.ckpt'
        model.save(ckpt_path, epoch, model_config=model_config)
        history_df = pd.DataFrame(history)
        history_df.to_csv(config['save_dir']+'/history.csv', index=False)
        break
        


    end = time.time()    
    if epoch==0:
        print ('epoch\ttrain_loss\ttest_loss\ttime (s)')
    print ("%s\t%.3f\t%.3f\t%.3f" %(epoch, train_loss, test_loss, end-st))
    # save model!
    # only save for the last epoch...

    if epoch == config['num_epochs']-1:
        ckpt_path = config['save_dir']+'/model_'+'.ckpt'
        model.save(ckpt_path, epoch, model_config=model_config)
        #save history as csv file
        history_df = pd.DataFrame(history)
        history_df.to_csv(config['save_dir']+'/history.csv', index=False)

