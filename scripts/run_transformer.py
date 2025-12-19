import pandas as pd
from torch import nn
from torch.utils.data import Subset
import matplotlib.pyplot as plt
from lstnn.dataset import get_dataset, PuzzleDataset
from lstnn.model import FFN, LSTM_combined, LSTM
import lstnn.transformer_main as transformer_main
from lstnn.seed import set_global_seed
import lstnn
from curricula import get_curriculum
import numpy as np
import torch
import time
import argparse
import os
import warnings


# number of threads to use
torch.set_num_threads(8)
parser = argparse.ArgumentParser('./main.py', description='Train RNN model on primitives dataset')
#parser.add_argument('--seed', type=int, default=1, help='seed (int) | default: 1')
parser.add_argument('--model_label', type=str, default='Transformer', help='model (str) | default: Transformer')
parser.add_argument('--seeds', nargs='+', type=int, help='<Required> Set flag, list of seeds', required=True)
parser.add_argument('--attnheads', type=int, default=4, help='attention heads per layer (int) | default: 1')
parser.add_argument('--layers', type=int, default=1, help='num transformer layer/blocks (int) | default: 1')
parser.add_argument('--pe', type=str, default='1dpe', help='positional encoding (str) | default: None (absolute 1d)')
parser.add_argument('--pe_init', type=float, default=1.0, help='initialization SD for learnable positional encodings | default: 1.0')
parser.add_argument('--embedding_dim', type=int, default=160, help='embedding dimensionality (int) | default: 1')
parser.add_argument('--training_acc_cutoff', type=float, default=0.0, help='when to stop training (btwn 0 and 1) | default: 0.99')
parser.add_argument('--lr', type=float, default=0.0001, help='weight decay (float)) | default: 0.01')
parser.add_argument('--device', type=str, default='cuda', help='device (str) | default: mps')
parser.add_argument('--col', type=int, default=0, help='')
parser.add_argument('--n_epochs', type=int, default=2000, help='')
parser.add_argument('--dropout', type=float, default=0.0, help='')
parser.add_argument('--wdecay', type=float, default=0.0, help='')
parser.add_argument('--curriculum', type=str, default='All', help='curriculum to train (str) | default: All')
parser.add_argument('--optimizer', type=str, default='AdamW', help='Optimizer (str) | default: AdamW')

# Example command:
# python expt3_splithalf_multidecoding.py --name='sim6' --device='cuda' --seed 1 --curriculum --weight_decay=0.0 --activity_norm=0.0 --split='random' --pe='rope'
# python expt3_splithalf_multidecoding.py --name='sim8' --device='cuda' --seed 1 --curriculum --weight_decay=0.0 --activity_norm=0.0 --split='newsymbol' --pe='rope'

def run(args):

    attnheads = args.attnheads
    nblocks = args.layers
    hidden_size = args.embedding_dim
    learning_rate = args.lr
    training_acc_cutoff = args.training_acc_cutoff
    seeds = args.seeds
    device = args.device
    device = torch.device(device)
    pe = args.pe
    pe_init = args.pe_init
    cutoff_length = args.col
    n_epochs = args.n_epochs
    dropout = args.dropout
    wdecay = args.wdecay
    model_label = args.model_label
    curriculum = args.curriculum
    optim = args.optimizer

    if pe_init!=1.0 and pe!='learn': 
        # altering pe_init only makes sense with pe=='learn'
        raise Exception("This argument doesn't make sense")

    if pe=='learn': 
        pestr = 'learn-' + str(pe_init)

    pe_dict = {'1dpe':'absolute',
               '2dpe':'absolute2d',
               'rope':'rope',
               'rope2':'rope2',
               'shaw':'relative',
               'nope':'nope',
               'cnope':'cnope',
               'scnope':'scnope',
               'rndpe':'rndpe',
               'rnd2':'rnd2',
               'learn0':'learn0',
               'learn':'learn',
               'clearn':'clearn',
               'learnu':'learnu'
               }
    
    # fixed parameters
    checkpoint_freq = 500 #epochs

    if optim == 'sgd':
        optim_str = 'sgd_'
    else:
        optim_str = ''

    # create results directory
    resultdir = f"/dccstor/synagi_compgen/LSTNN/results/" \
                f"model-{model_label}_{optim_str}" \
                f"pe-{pestr}_" \
                f"nl-{nblocks}_" \
                f"do-{dropout}_" \
                f"wd-{wdecay}_" \
                f"at-{attnheads}_" \
                f"hs-{hidden_size}_" \
                f"curr-{curriculum}_" \
                f"lr-{learning_rate}_" \
                f"co-{training_acc_cutoff}_" \
                f"col-{cutoff_length}/"
    os.makedirs(resultdir, exist_ok = True) 
    
    grid_size = 4 * 4  # latin squares dimensions (number of tokens)
    input_dim = 5 # 5 possible input codes
    output_size = 4         # 4 possible motor responses

    training_files = ['../data/nn/generated_puzzle_data_binary_dist80.csv',
                    '../data/nn/generated_puzzle_data_ternary_dist80.csv',
                    '../data/nn/generated_puzzle_data_quaternary_dist80.csv']
    validation_file = '../data/nn/puzzle_data_original.csv'

    # size of minibatches for training and testing
    # for testing there is only 108 total
    train_batch_size = 128
    valid_batch_size = 108

    for seed in seeds:
        set_global_seed(seed)

        if model_label == 'Transformer':
            model = transformer_main.Transformer(input_dim=input_dim,
                        output_dim=output_size,
                        max_tokens=grid_size,
                        nhead=attnheads,
                        nblocks=nblocks,
                        embedding_dim=hidden_size,
                        dropout=dropout,
                        positional_encoding=pe_dict[pe],
                        pe_init=pe_init)

        # define the loss and optimiser
        if optim == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        elif optim == 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=wdecay)
        else:
            raise Exception('Wrong optimizer')
        loss_fn = nn.CrossEntropyLoss()

        # start training
        df = pd.DataFrame(columns=['epoch', 'block', 'loss',
                                    'accuracy', 'condition', 'phase'])

        # create the datasets and data loaders
        train_index, test_index, train_df = get_curriculum(
            curriculum, training_files)

        train_dataloader = torch.utils.data.DataLoader(Subset(get_dataset(
            training_files), train_index), batch_size=train_batch_size, shuffle=True)

        test_dataloader = torch.utils.data.DataLoader(Subset(get_dataset(
            training_files), test_index), batch_size=train_batch_size, shuffle=True)

        valid_dataloader = torch.utils.data.DataLoader(get_dataset(
            validation_file), batch_size=valid_batch_size, shuffle=False)

        # train until cri.dateria is reached
        epoch = 0
        mean_train_correct = 0
        cutoff_satisfied = 0
        start = time.time()
        #while cutoff_satisfied < cutoff_length:
        while epoch <= n_epochs:

            # load checkpoint, if it exists
            out = f"s-{seed}_" \
                  f"e-{epoch}" 
            if os.path.isfile(resultdir + out + '.pt'):
                next_checkpt = f"s-{seed}_e-{epoch+checkpoint_freq}"
                if os.path.isfile(resultdir + next_checkpt + '.pt'):
                    epoch += checkpoint_freq
                    print(resultdir + out, 'checkpoint exists, skipping to epoch', epoch)
                    continue
                else:
                    model.load_state_dict(torch.load(resultdir + out + '.pt',map_location=device))
                    epoch += 1
                    print(resultdir + out, 'checkpoint exists, skipping to epoch', epoch)

            # training
            results, model = train(train_dataloader,
                                    model,
                                    loss_fn,
                                    optimizer,
                                    device=device)

            # update mean training accuracy and cutoff
            mean_train_correct = results['train_acc']['total'].copy()
            if mean_train_correct > training_acc_cutoff:
                cutoff_satisfied += 1
            else:  # reset the cutoff
                cutoff_satisfied = 0

            if epoch % 10 == 0:
                end = time.time()
                print('Epoch ', epoch,
                    ': Train acc = ', np.round(mean_train_correct, 3),
                    ', time = ', np.round(end-start, 3)
                )
                #      ', Test acc = ', np.round(mean_test_correct, 3),
                #      ', Val. acc = ', np.round(mean_valid_correct, 3),
                #      )
                start = time.time()

            if epoch % checkpoint_freq == 0:
                # evaluate performance and save
                # testing
                results['test_loss'], results['test_acc'] = evaluate_model(
                    test_dataloader, model, loss_fn, device)
                mean_test_correct = results['test_acc']['total'].copy()

                # validating
                results['validation_loss'], results['validation_acc'] = evaluate_model(
                    valid_dataloader, model, loss_fn, device)
                mean_valid_correct = results['validation_acc']['total'].copy()

                print('Epoch ', epoch,
                    ': Train acc = ', np.round(mean_train_correct, 3),
                    ', Test acc = ', np.round(mean_test_correct, 3),
                    ', Val. acc = ', np.round(mean_valid_correct, 3),
                    ', time = ', np.round(end-start, 3)
                    )
                # update results df
                df = results_to_df(df, results, epoch, block=0)

                # save model
                out = f"s-{seed}_" \
                      f"e-{epoch}" 
                print(resultdir + out)
                torch.save(model.state_dict(), resultdir + out+'.pt')
                df.to_csv(resultdir + out+'.csv', index=False)

            # update epoch
            epoch += 1




def train(dataloader, model, loss_fn, optimizer, device='cpu'):

    # define device
    device = torch.device(device)
    model.to(device)

    # define conditions
    if type(dataloader.dataset) is torch.utils.data.dataset.Subset:
        conditions = list(np.array(dataloader.dataset.dataset.conditions)[
            dataloader.dataset.indices])
    elif type(dataloader.dataset) is src.dataset.PuzzleDataset:
        conditions = dataloader.dataset.conditions

    # Placeholders to save the loss and accuracy at each iteration
    train_loss = []
    train_acc = {'total': []}
    for condition in np.unique(conditions):
        train_acc[condition.lower()] = []

    for i, batch in enumerate(dataloader):

        # get features
        train_features, train_labels, index = batch[0], batch[1], batch[2]
        # flatten to accommodate transformer
        train_features = torch.flatten(train_features,start_dim=1,end_dim=2)
        #
        train_features = train_features.to(device)
        train_labels = train_labels.to(device)

        # Compute prediction and loss
        if model.positional_encoding=='scnope':
            # currently using temperature as sparsification
            l1_lambda = 0.0
            out, l1_reg = model(train_features)
            loss = loss_fn(out, train_labels) + (l1_reg * l1_lambda) # sparsity
            #print('loss',loss, ' | l1 loss', l1_reg*l1_lambda)
        else:
            out = model(train_features)
            loss = loss_fn(out, train_labels) 

        # Backpropagation
        optimizer.zero_grad()  # clear previous gradients
        loss.backward()        # compute gradients
        optimizer.step()       # update weights

        # compute task accuracy
        accuracy = torch.sum(torch.argmax(out, dim=1) == torch.argmax(
            train_labels, dim=1)).item() / train_labels.shape[0]

        # Store current values of loss and accuracy
        train_loss.append(loss.item())
        train_acc['total'].append(accuracy)

        # calculate accuracy per condition
        acc_trial = torch.argmax(out, dim=1) == torch.argmax(
            train_labels, dim=1)

        for condition in np.unique(batch[3]):
            avg_acc = np.mean(acc_trial.cpu().numpy()[
                              np.array(batch[3]) == condition])
            train_acc[condition.lower()].append(avg_acc)

    # calc averages
    avg_train_acc = {key: np.mean(train_acc[key]) for key in train_acc}
    avg_train_loss = np.mean(train_loss)

    return {'train_loss': avg_train_loss, 'train_acc': avg_train_acc}, model


def evaluate_model(dataloader, model, loss_fn, device='cpu'):

    # define conditions
    if type(dataloader.dataset) is torch.utils.data.dataset.Subset:
        conditions = list(np.array(dataloader.dataset.dataset.conditions)[
            dataloader.dataset.indices])
    elif type(dataloader.dataset) is PuzzleDataset:
        conditions = dataloader.dataset.conditions

    # Placeholders to save the loss and accuracy at each iteration
    test_loss = []
    test_acc = {'total': []}
    for condition in np.unique(conditions):
        test_acc[condition.lower()] = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):

            # get features
            test_features, test_labels, index = batch[0], batch[1], batch[2]
            # flatten to accommodate transformer
            test_features = torch.flatten(test_features,start_dim=1,end_dim=2)
            #
            test_features = test_features.to(device)
            test_labels = test_labels.to(device)

            # Compute prediction and loss
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if model.positional_encoding=='scnope':
                    out, l1_reg = model(test_features)
                else:
                    out = model(test_features)
                loss = loss_fn(out, test_labels)

            # compute task accuracy
            accuracy = torch.sum(torch.argmax(out, dim=1) == torch.argmax(
                test_labels, dim=1)).item() / test_labels.shape[0]

            # Store current values of loss and accuracy
            test_loss.append(loss.item())
            test_acc['total'].append(accuracy)

            # calculate accuracy per condition
            acc_trial = torch.argmax(out, dim=1) == torch.argmax(
                test_labels, dim=1)

            for condition in np.unique(batch[3]):
                avg_acc = np.mean(acc_trial.cpu().numpy()[
                                  np.array(batch[3]) == condition])
                test_acc[condition.lower()].append(avg_acc)

    # calc averages
    avg_test_acc = {key: np.mean(test_acc[key]) for key in test_acc}
    avg_test_loss = np.mean(test_loss)
    return avg_test_loss, avg_test_acc


def results_to_df(df, results, epoch, block):
    # little helper function to save results
    # to df
    for phase in ['train', 'test', 'validation']:
        # save out total scores
        row = pd.DataFrame({'epoch': [epoch],
                            'block': [block],
                            'loss': results[phase+'_loss'],
                            'accuracy': results[phase+'_acc']['total'],
                            'condition': 'average',
                            'phase': phase})
        df = pd.concat([df, row], ignore_index=True)

        # dynamically get conditions as training
        # phase may only include certain conds
        conds = list(results[phase+'_acc'].keys())
        conds.remove('total')
        for condition in conds:
            row = pd.DataFrame({'epoch': [epoch],
                                'block': [block],
                                'loss': np.nan,
                                'accuracy': results[phase+'_acc'][condition],
                                'condition': condition,
                                'phase': phase})
            df = pd.concat([df, row], ignore_index=True)
    return df

if __name__ == '__main__':
    args = parser.parse_args()
    run(args)

