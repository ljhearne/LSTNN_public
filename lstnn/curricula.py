import pandas as pd
from sklearn.model_selection import train_test_split
from lstnn.dataset import puzzle_csv_to_df


def get_curriculum(c, puzzle_files, test_size=0.1, random_state=42):
    '''
    c = what curriculum?
    puzzle_files = list of csvs for binary, ternary and quaternary

    Curricula:
    - A represents training and testing on all three types of problems

    '''
    # create a dataframe of all the training files
    df = pd.DataFrame()
    for f in puzzle_files:
        df = pd.concat([df, puzzle_csv_to_df(f)])
    df = df.reset_index()

    if c == 'All':
        # perform split stratified by condition
        x = range(len(df))
        y = df['condition'].to_list()

        train_index, test_index, _, _ = train_test_split(
            x, y, stratify=y, test_size=test_size, random_state=random_state)

    elif c == 'Binary':
        train_index = df[df['condition'] == 'Binary'].index.values
        test_index = df[df['condition'] != 'Binary'].index.values

    elif c == 'NotQuaternary':
        train_index = df[df['condition'] != 'Quaternary'].index.values
        test_index = df[df['condition'] == 'Quaternary'].index.values

    elif c == 'BinaryAndQuaternary':
        train_index = df[df['condition'] != 'Ternary'].index.values
        test_index = df[df['condition'] == 'Ternary'].index.values

    elif c == 'Ternary':
        train_index = df[df['condition'] == 'Ternary'].index.values
        test_index = df[df['condition'] != 'Ternary'].index.values

    elif c == 'TernaryAndQuaternary':
        train_index = df[df['condition'] != 'Binary'].index.values
        test_index = df[df['condition'] == 'Binary'].index.values

    elif c == 'Quaternary':
        train_index = df[df['condition'] == 'Quaternary'].index.values
        test_index = df[df['condition'] != 'Quaternary'].index.values

    elif c == 'Solution1':
        train_index = df[df['solutions'] != '1'].index.values
        test_index = df[df['solutions'] == '1'].index.values

    else:
        print("That curriculum doesn't exist")

    return train_index, test_index, df
