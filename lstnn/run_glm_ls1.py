import nibabel as nb
import numpy as np
import pandas as pd
import xarray as xr
import json
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.glm.first_level import run_glm
from nilearn.glm.contrasts import compute_contrast
from nilearn.plotting import plot_design_matrix
import argparse
from tqdm import tqdm

# Create parser for options
parser = argparse.ArgumentParser(
    description='''Run nilearn based fMRI GLM for the LST''')

# These parameters must be passed to the function
parser.add_argument('--inputs',
                    type=str,
                    nargs=3,
                    default=None,
                    help='''input bold dat, 3 strings''')

parser.add_argument('--events',
                    type=str,
                    nargs=3,
                    default=None,
                    help='''input BIDS events file (.tsv)''')

parser.add_argument('--bold_json',
                    type=str,
                    default=None,
                    help='''BIDS json file to gather TR''')

parser.add_argument('--model',
                    type=str,
                    default=None,
                    help='''model to use for GLM''')

parser.add_argument('--output',
                    type=str,
                    default=None,
                    help='''output file (xxx.nc)''')


def run_glm_ls1(inputs, events, bold_json, model, output):

    # check inputs
    assert len(inputs) == 3, "3 Runs not found"
    assert len(events) == 3, "3 Events not found"

    # get tr information
    tr = json.load(open(bold_json,))['RepetitionTime']

    # get the bold data and calculate frames
    # load the first to calculate shape
    n_TR_per_run, n_vertex = nb.load(inputs[0]).get_fdata().shape
    bold_array = np.zeros((1, n_vertex))
    for run in range(len(inputs)):
        bold_array = np.vstack((bold_array, nb.load(inputs[run]).get_fdata()))
    bold_array = np.delete(bold_array, 0, axis=0)  # remove the init empty row
    assert bold_array.shape[0] == 3750, "Data wrong shape!"
    
    # frame times: assumes a slice ref of 0.5
    frame_times = tr * (np.arange(bold_array.shape[0]) + .5)

    # use nilearn to build design matrix with each trial as
    # an independent regressor
    design_matrix = np.zeros((bold_array.shape[0], 1))

    # get events from BIDS and account for n runs in the onset time
    df = pd.DataFrame()
    for run in range(len(events)):
        _df = pd.read_csv(events[run], delimiter='\t')
        _df['onset'] = _df['onset'] + (run * (tr*n_TR_per_run))
        df = pd.concat([df, _df])

    # assert possible errors in the df
    assert df['LST_id_orig'].nunique() == 48, "Error in events df - 48 puzzles"

    for i in range(1, 49):
        assert sum(df['LST_id_orig'] ==
                   i) == 3, "Error in events df - 3 trials per"

    # convert to a nilearn df
    nilearn_events = pd.DataFrame(columns=['onset', 'trial_type'])
    # nilearn_events['onset'] = df['onset'].copy()
    # nilearn_events['duration'] = 5
    nilearn_events['onset'] = df['onset'].copy() +2 # "late stage processing"
    nilearn_events['duration'] = 3
    #nilearn_events['trial_type'] = 'LST'+df['LST_id_orig'].astype('str')
    nilearn_events['trial_type'] = 'LST'+df['LST_id'].astype('str')

    # Begin the LS1 model loop
    # Loop through each of the 48 events and split the conditions into
    # i) trial of interest and ii) everything else

    # generate a list of LST items that matches the events_df
    puzzle_list = ['LST' + str(i) for i in range(1, 145)]

    # preallocate final result arrays
    # xarray allows for better indexing, e.g., stat_array.sel(puzzle='LST10', stat='z')
    stat_array = xr.DataArray(np.zeros((bold_array.shape[1], len(puzzle_list), 3)),
                              dims=('vertex', 'puzzle', 'stat'),
                              coords={'stat': ['beta', 't', 'z'],
                                      'puzzle': puzzle_list})

    for i, trial in tqdm(enumerate(puzzle_list)):

        # convert events to current vs. all other trials
        ls_events = nilearn_events.copy()
        ls_events.loc[ls_events.trial_type != trial,
                      'trial_type'] = 'all_other_trials'
        
        # create the design matrix
        design_matrix = make_first_level_design_matrix(frame_times,
                                                       events=ls_events,
                                                       hrf_model='spm',
                                                       drift_model=None)
        
        # add session columns
        for run in range(len(events)):
            s = 0 + (n_TR_per_run * run)
            e = (n_TR_per_run * run) + n_TR_per_run
            data = np.zeros(design_matrix.shape[0])
            data[s:e] = 1
            design_matrix['session'+str(int(run+1))] = data

        # run the glm
        labels, estimates = run_glm(bold_array, design_matrix.values, n_jobs=1)

        # compute contrasts for the trial of interest
        col_index = [i for i, j in enumerate(
            design_matrix.columns) if j.startswith('LST')]
        contrast_matrix = np.eye(design_matrix.shape[1])
        basic_contrasts = dict([(column, contrast_matrix[i])
                                for i, column in enumerate(design_matrix.columns[col_index])])

        contrast = compute_contrast(
            labels, estimates, basic_contrasts[trial], stat_type='t')
        stat_array[:, i, 0] = contrast.effect_size().flatten()
        stat_array[:, i, 1] = contrast.stat().flatten()
        stat_array[:, i, 2] = contrast.z_score().flatten()

    # save outputs
    stat_array.to_netcdf(output)
    plot_design_matrix(design_matrix, output_file=output+'.png')
    return stat_array


if __name__ == '__main__':
    # Read in user-specified parameters
    args = parser.parse_args()

    # run the glm
    run_glm_ls1(args.inputs, args.events,
                args.bold_json, args.model, args.output)
