from model import CVAE
from utils import *
import numpy as np
import time
import pandas as pd
import torch
from copy import deepcopy


# Single source of truth for run configuration.
# Edit values here directly; CLI arguments are intentionally disabled.
config = {
    'batch_size': 128,
    'latent_size': 200, # latent vector size; also Transformer token embedding size
    'unit_size': 512, # hidden size (LSTM) / internal Transformer d_model width
    'n_rnn_layer': 2, # number of RNN layers in the encoder and decoder
    'seq_length': 120, # maximum length of the input and output sequences (smiles strings)
    'prop_file': 'prop_mw_logp.txt',
    'mean': 0.0,
    'stddev': 1.0,
    'num_epochs': 100,
    'lr': 0.0001,
    'num_prop': None,
    'save_dir': 'save/',
    'save_every': 10, # save a checkpoint every N epochs
    'patientce': 10,
    'early_stopping_patience': 10,
    'early_stopping_min_delta': 0.0,
    'early_stopping_restore_best': True,
    'optimizer': 'adamw',  # 'adam' or 'adamw'
    'weight_decay': 0.01,
    'use_reduce_lr_on_plateau': True,
    'lr_plateau_factor': 0.5,
    'lr_plateau_patience': 5,
    'lr_plateau_threshold': 1e-4,
    'lr_plateau_min_lr': 1e-6,
    'model_mode': 'transformer',  # 'lstm' or 'transformer'
    'transformer_heads': 8, # number of heads in the multi-head attention mechanism
    'transformer_ff_size': 768, # dimension of the feedforward network in the transformer layers
    'transformer_dropout': 0.15, # dropout rate for the transformer layers
    'train_ratio': 0.75, # ratio of data to use for training (the rest is used for testing)
}

# convert config dict to a dataclass for better attribute access and type checking
config = compose_train_config_from_dict(config)


print (config)
#convert smiles to numpy array
# we will have two version of output, one with start and end token, one without. the one with start and end token is used for training, the one without is used for testing.
molecules_input, molecules_output, char, vocab, labels, length = load_data(config['prop_file'], config['seq_length'])
vocab_size = len(char)
if labels.ndim != 2 or labels.shape[1] == 0:
    raise ValueError('Property file must contain at least one numeric conditioning column after SMILES.')
config['num_prop'] = int(labels.shape[1])
print(f'inferred num_prop from {config["prop_file"]}: {config["num_prop"]}')
model_config = get_model_config(config, vocab_size=vocab_size)

#make save_dir
ensure_dir(config['save_dir'])

# save a single source of truth for recreating the trained model
training_config_path = save_training_config(model_config, config['save_dir'])
print(f'saved training config to: {training_config_path}')

#divide data into training and test set
# can leak...
# could look into scaffold splitting for this!
train_molecules_input, test_molecules_input = split_train_test(molecules_input, config['train_ratio'])
train_molecules_output, test_molecules_output = split_train_test(molecules_output, config['train_ratio'])
train_labels, test_labels = split_train_test(labels, config['train_ratio'])
train_length, test_length = split_train_test(length, config['train_ratio'])

model = CVAE(vocab_size, model_config)
print('Number of parameters : ', sum(p.numel() for p in model.parameters() if p.requires_grad))

scheduler = None
if config['use_reduce_lr_on_plateau']:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        model.optimizer,
        mode='min',
        factor=config['lr_plateau_factor'],
        patience=config['lr_plateau_patience'],
        threshold=config['lr_plateau_threshold'],
        min_lr=config['lr_plateau_min_lr'],
    )

history = {
    'train_loss': [],
    'test_loss': [],
    'lr': [],
}

best_test_loss = float('inf')
best_epoch = -1
epochs_without_improvement = 0
best_state_dict = None

start_time = time.time()

for epoch in range(config['num_epochs']):

    st = time.time()
    # Learning rate scheduling 
    #model.assign_lr(learning_rate * (decay_rate ** epoch))
    # lr scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
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

    if not np.isfinite(train_loss) or not np.isfinite(test_loss):
        print(f'non-finite loss detected at epoch {epoch} (train={train_loss}, test={test_loss}), stopping early')
        break

    if scheduler is not None:
        scheduler.step(float(test_loss))

    current_lr = model.optimizer.param_groups[0]['lr']

    history['train_loss'].append(train_loss)
    history['test_loss'].append(test_loss)
    history['lr'].append(current_lr)

    # robust early stopping with min_delta
    improved = float(test_loss) < (best_test_loss - float(config['early_stopping_min_delta']))
    if improved:
        best_test_loss = float(test_loss)
        best_epoch = epoch
        epochs_without_improvement = 0
        if config['early_stopping_restore_best']:
            best_state_dict = deepcopy(model.state_dict())
    else:
        epochs_without_improvement += 1

    #early stopping if no improvment for a while
    # 
    if epochs_without_improvement >= config['early_stopping_patience']:
        print(
            f'early stop at epoch {epoch} since no improvement for '
            f'{epochs_without_improvement} epochs (best epoch: {best_epoch}, best test loss: {best_test_loss:.6f})'
        )

        if config['early_stopping_restore_best'] and best_state_dict is not None:
            model.load_state_dict(best_state_dict)
            print(f'restored best model weights from epoch {best_epoch}')
        
        ckpt_path = config['save_dir']+'/model_'+str(epoch)+'.ckpt'
        model.save(ckpt_path, epoch, model_config=model_config)
        history_df = pd.DataFrame(history)
        history_df.to_csv(config['save_dir']+'/history.csv', index=False)
        break
        


    end = time.time()    
    if epoch==0:
        print ('epoch\ttrain_loss\ttest_loss\tlr\ttime (s)')
    #print ("%s\t%.3f\t%.3f\t%.6f\t%.3f" %(epoch, train_loss, test_loss, current_lr, end-st))

    passed_time = end - st
    
    # logic to calculate expected time remaining, based on time taken and epochs
    
    if epoch > 0:
        time_per_epoch = (end - start_time) / (epoch + 1)
        expected_time_remaining = time_per_epoch * (config['num_epochs'] - epoch - 1)
        print(f'{epoch}\t{train_loss:.3f}\t{test_loss:.3f}\t{current_lr:.6f}\t{passed_time:.3f}\tETA: {expected_time_remaining/60:.2f} min')

   

    # save model!
    # only save for the last epoch...

    if epoch == config['num_epochs']-1 or (epoch + 1) % config['save_every'] == 0:
        if config['early_stopping_restore_best'] and best_state_dict is not None:
            model.load_state_dict(best_state_dict)
            print(f'training ended; restored best model weights from epoch {best_epoch}')
        ckpt_path = config['save_dir']+'/model_'+str(epoch)+'.ckpt'
        model.save(ckpt_path, epoch, model_config=model_config)
        #save history as csv file
        history_df = pd.DataFrame(history)
        history_df.to_csv(config['save_dir']+'/history.csv', index=False)

