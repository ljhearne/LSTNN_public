from random import choice
import pandas as pd
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


class PuzzleDataset(Dataset):
    '''Custom dataset with indices'''

    def __init__(self, puzzles, solutions, conditions):
        self.puzzles = puzzles
        self.solutions = solutions
        self.conditions = conditions

    def __getitem__(self, index):
        data = self.puzzles[index, :, :, :]
        target = self.solutions[index]
        condition = self.conditions[index]
        return data, target, index, condition

    def __len__(self):
        return len(self.puzzles)

    def print_puzzle(self, index):
        v, I = torch.max(self.puzzles[index], 2)
        return ((v > 0).long()*(I+1)).squeeze(), torch.where(self.solutions[index])[0][0]+1


def puzzle_csv_to_df(csv_files):
    if type(csv_files) == str:
        csv_files = [csv_files]

    df = pd.DataFrame()
    for csv in csv_files:
        df = pd.concat([df, pd.read_csv(csv, dtype=str)], ignore_index=True)

    # replace the IDs with the index
    df['ID'] = df.index.copy()
    return df


def puzzles_to_tensor(csv_files=['puzzle_data_original.csv']):
    """Converts LST puzzles from csv to one hot encoded tensors."""

    # get puzzle csv as df
    df = puzzle_csv_to_df(csv_files)
    puzzle_grid = df_puzzles_to_grid(df)

    # store solutions as an array
    solutions = string_to_array(df.solutions.values)

    # convert puzzles to one hot encoded tensor
    puzzles_one_hot = np.zeros((len(df), 4, 4, 5))
    for trial in range(len(df)):
        puzzles_one_hot[trial, :, :, :] = grid_to_one_hot(
            puzzle_grid[:, :, trial])
    puzzles_one_hot = torch.from_numpy(puzzles_one_hot).float()

    # convert solutions
    solutions_one_hot = np.zeros((len(df), 4))
    for trial in range(len(df)):
        for channel, stimuli in enumerate([1, 2, 3, 4]):
            solutions_one_hot[trial, channel] = solutions[trial] == stimuli
    solutions_one_hot = torch.from_numpy(solutions_one_hot).float()

    return puzzles_one_hot, solutions_one_hot


def convert_puzzle_data():
    '''
    Transcribes the original LST fMRI task data from .mat
    into a more readable .csv file. This code only needs to
    be run once.

    The puzzle is represented as 16 continuous numbers that
    index into the LST like so:

    -------------
    | 0| 1| 2| 3|
    -------------
    | 4| 5| 6| 7|
    -------------
    | 8| 9|10|11|
    -------------
    |12|13|14|15|
    -------------
    The numbers 1, 2, 3 and 4 each represent a different shape
    and '9' represents the target (a question mark in the fMRI)
    experiment. In this code the target ('9') gets converted to
    a 5 to allow easier one hot encoding.

    LST_num represents the original LST number each of the
    puzzles are a copy of. These were rotated three times each
    in the experiment. So, LST puzzle 0, 1, 2 are 'geometrically'
    identical, just rotated 90 degrees.
    '''

    task_data_original = loadmat('LST_data.mat')['taskData']

    df = pd.DataFrame(
        columns=['ID', 'LST_num', 'condition', 'puzzles', 'solutions'])

    for trial in range(108):

        # LST number
        LST_num = int(np.floor((trial / 3) + 1))

        # Condition
        if LST_num < 13:
            condition = 'Binary'

        elif LST_num > 24:
            condition = 'Quaternary'

        else:
            condition = 'Ternary'

        # Puzzle
        puzzle_data = str(task_data_original[trial, 0:-1])
        for char in ['[', ']', ' ']:
            puzzle_data = puzzle_data.replace(char, '')

        # replace target '9' with '5'
        for char in ['9']:
            puzzle_data = puzzle_data.replace(char, '5')

        # Solution
        solution = str(task_data_original[trial, -1])

        # correct typo in original dataset
        if trial == 99:
            solution = '1'
        if trial == 100:
            solution = '2'
        if trial == 101:
            solution = '3'

        # add to the dataframe
        df.loc[trial] = [trial, LST_num, condition, puzzle_data, solution]

    df.to_csv('puzzle_data_original.csv', index=False)


def sudoku_to_LST(npz_file='sudokuLST.npz'):

    input_data = np.load(npz_file)

    puzzles = input_data['quizzes']
    solutions = input_data['solutions']

    # create random targets and solutions
    n_puzzles = puzzles.shape[0]
    new_solutions = np.zeros((n_puzzles))
    new_puzzles = puzzles.copy()

    for p in range(n_puzzles):

        # get individual puzzle
        puzzle = puzzles[p, :, :].reshape(16).copy()

        # choose a target from a blank
        blanks = np.where(puzzle == 0)[0]
        target = choice(blanks)
        puzzle[target] = 5

        # put back in puzzles
        puzzle = puzzle.reshape(4, 4)
        new_puzzles[p, :, :] = puzzle.copy()

        # note the correct solution
        solution = solutions[p, :, :].reshape(16).copy()[target]
        new_solutions[p] = solution.copy()

    # save in a consistent dataframe format
    df = pd.DataFrame(
        columns=['ID', 'LST_num', 'condition', 'puzzles', 'solutions'])

    for trial in range(n_puzzles):

        # Puzzle
        puzzle_data = str(new_puzzles[trial, :, :].reshape(16))
        for char in ['[', ']', ' ']:
            puzzle_data = puzzle_data.replace(char, '')
        # Solution
        solution = str(int(new_solutions[trial]))

        # add to the dataframe
        df.loc[trial] = [trial, np.nan, np.nan, puzzle_data, solution]

    df.to_csv('generated_puzzle_data.csv', index=False)


def sudoku_to_tensor(npz_file='sudokuLST.npz'):
    '''
    Converts the automated sudoku data
    '''

    def sudoku_onehot(input):
        # convert puzzles to one hot encoded tensor
        onehot = np.zeros((len(input), 4, 4, 4))
        for trial in range(len(input)):
            for channel, stimuli in enumerate(range(1, 5)):
                onehot[trial, :, :, channel] = input[trial, :, :] == stimuli
        onehot = torch.from_numpy(onehot).float()
        return onehot

    input_data = np.load(npz_file)

    puzzles = input_data['quizzes']
    puzzles_onehot = sudoku_onehot(puzzles)

    solutions = input_data['solutions']
    solutions_onehot = sudoku_onehot(solutions)

    return puzzles_onehot, solutions_onehot


def string_to_array(input_string):
    list = [int(char) for char in input_string]
    return np.array(list)


def df_puzzles_to_grid(df):
    # read in, reshape puzzle and store in new
    # 4 x 4 x length array
    puzzle_grid = np.zeros((4, 4, len(df)))
    for i, trial in df.iterrows():
        puzzle = string_to_array(trial.puzzles)
        puzzle_grid[:, :, i] = puzzle.reshape(4, 4)
    return puzzle_grid


def grid_to_one_hot(puzzle_grid):
    # store a 4 x 4 grid into a one hot encoded
    # 4 x 4 x 5 (channel) array
    puzzle_one_hot = np.zeros((4, 4, 5))
    for channel, stimuli in enumerate(range(1, 6)):
        puzzle_one_hot[:, :, channel] = puzzle_grid == stimuli
    return puzzle_one_hot


def get_dataset(csv_files=['puzzle_data_original.csv']):

    # get one hot encoded puzzles and solutions
    puzzles, solutions = puzzles_to_tensor(csv_files)

    # get attached conditions
    df = pd.DataFrame()

    if type(csv_files) == str:
        csv_files = [csv_files]

    for f in csv_files:
        df = pd.concat([df, puzzle_csv_to_df(f)])

    return PuzzleDataset(puzzles, solutions, df.condition.to_list())
