import numpy as np
import pandas as pd
import src.dataset as ds
from src.verify_LST import verify_LST
from src.denoise import _derivative


def test_training_solutions():
    # test the all training data is solveable and
    # that solutions match estimated solutions and
    # that conditions match estimated conditions
    for dist in ['60']:
        for condition in ['binary', 'ternary', 'quaternary']:
            dataset = ds.get_dataset(
                ['../data/nn/generated_puzzle_data_'+condition+'_dist'+dist+'.csv'])

            for trial in range(len(dataset)):
                puzzle, solution = dataset.print_puzzle(trial)
                est_condition, est_solution, _ = verify_LST(puzzle.numpy())
                assert solution.numpy() == est_solution, "Error: solutions do not match"
                assert condition == est_condition.lower(), "Error: conditions do not match"

# def test_model_Reproducibility():
    # test that pytorch models produce same results
    # with the same seed


def test_derivatives():
    # hard coded, just needs a single subject's confound file.
    confounds = '/home/lukeh/hpcworking/shared/projects/LST7T/data/derivatives/fmriprep/sub-02/func/sub-02_task-rest_run-2_desc-confounds_timeseries.tsv'
    conf_df = pd.read_csv(confounds, delimiter="\t")
    a = conf_df['rot_x'].values

    # derivative1
    b = conf_df['rot_x_derivative1'].values
    c = _derivative(a)
    r = np.corrcoef(b[1:],c[1:])[0,1]
    assert r > 0.99, "derivative equality assertion failed"

    # power2
    b = conf_df['rot_x_power2'].values
    c = a*a
    r = np.corrcoef(b,c)[0,1]
    assert r > 0.99, "power equality assertion failed"

    # derivative1_power2
    b = conf_df['rot_x_derivative1_power2'].values
    c = _derivative(a)*_derivative(a)
    r = np.corrcoef(b[1:],c[1:])[0,1]
    assert r > 0.99, "derivative_power equality assertion failed"