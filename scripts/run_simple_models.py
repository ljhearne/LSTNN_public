import torch
import pandas as pd
from torch import nn
from torch.utils.data import Subset
from lstnn.seed import set_global_seed
from lstnn.curricula import get_curriculum
from lstnn.dataset import get_dataset, PuzzleDataset
from lstnn.ffn_main import FFN
from lstnn.lstm_main import LSTM
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int)
parser.add_argument('--model_label', type=str)
parser.add_argument('--device', type=str)

# Params
torch.set_num_threads(4)
n_epochs = 4000
checkpoint_freq = 200
curriculum = 'All'
training_files = ['/home/lukeh/projects/LSTNN/data/nn/generated_puzzle_data_binary_dist80.csv',
                  '/home/lukeh/projects/LSTNN/data/nn/generated_puzzle_data_ternary_dist80.csv',
                  '/home/lukeh/projects/LSTNN/data/nn/generated_puzzle_data_quaternary_dist80.csv']
validation_file = '/home/lukeh/projects/LSTNN/data/nn/puzzle_data_original.csv'
train_batch_size = 128
valid_batch_size = 108
learning_rate = 0.0001


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
            test_features = test_features.to(device)
            test_labels = test_labels.to(device)

            # Compute prediction and loss
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


def train(dataloader, model, loss_fn, optimizer, device='cpu'):

    # define device
    device = torch.device(device)
    model.to(device)

    # define conditions
    if type(dataloader.dataset) is torch.utils.data.dataset.Subset:
        conditions = list(np.array(dataloader.dataset.dataset.conditions)[
            dataloader.dataset.indices])
    elif type(dataloader.dataset) is lstnn.dataset.PuzzleDataset:
        conditions = dataloader.dataset.conditions

    # Placeholders to save the loss and accuracy at each iteration
    train_loss = []
    train_acc = {'total': []}
    for condition in np.unique(conditions):
        train_acc[condition.lower()] = []

    for i, batch in enumerate(dataloader):

        # get features
        train_features, train_labels, index = batch[0], batch[1], batch[2]
        train_features = train_features.to(device)
        train_labels = train_labels.to(device)

        # Compute prediction and loss
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


def run(model_label, seed, device, hs=160):

    hs = 2560
    # Init. seed
    set_global_seed(seed)

    # organise the output
    resultdir = f"/home/lukeh/projects/LSTNN/results/" \
                f"model-{model_label}/" \
                f"nl-4_" \
                f"do-0_" \
                f"wd-0_" \
                f"hs-{hs}_" \
                f"lr-{learning_rate}_"

    # Initialise the model to be used
    if model_label == 'ffn':
        model = FFN(hidden_size=hs)

    elif model_label == 'lstm':
        model = LSTM(bidirectional=True)
        resultdir = resultdir+"bidirectionalTrue_"
        
    else:
        print('Not a model')

    # define the loss and optimiser
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=learning_rate,
                                  weight_decay=0)
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
    start = time.time()
    for epoch in range(n_epochs+1):

        # training
        results, model = train(train_dataloader,
                               model,
                               loss_fn,
                               optimizer,
                               device=device)

        # update mean training accuracy and cutoff
        mean_train_correct = results['train_acc']['total'].copy()
        end = time.time()
        if epoch % checkpoint_freq == 0:  # save results every _ epochs

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

        elif epoch % 10 == 0:  # print results every 100 epochs

            print('Epoch ', epoch,
                  ': Train acc = ', np.round(mean_train_correct, 3),
                  ', time = ', np.round(end-start, 3)
                  )


if __name__ == '__main__':
    args = parser.parse_args()
    run(args.model_label, args.seed, args.device)
