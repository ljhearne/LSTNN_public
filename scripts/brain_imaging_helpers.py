import xarray as xr
import numpy as np
from tqdm import tqdm
from lstnn.parcellation import Parcellation
import rsatoolbox.data as rsd
import pandas as pd
from os.path import isfile
# from bids import BIDSLayout, BIDSLayoutIndexer


def load_single_subject(subj, stat, cortex, cortex_res, scale, denoise):

    # load LS GLM file with 144 trials
    subj_str = str(subj).zfill(2)
    glm_in = f"../data/fMRI/fmriprep_denoised_glm_ls1/sub-{subj_str}/sub-{subj_str}_task-LST_space-fsLR_denoise-{denoise}_model-144_glm.nc"

    # combine into an xarray dataset
    glm_ds = xr.open_mfdataset(glm_in, concat_dim='subject', combine='nested')

    # reorder all the puzzles from Binary -> Ternary -> Quaternary -> Null
    # and select the specified stat
    puzzle_list = ['LST' + str(i) for i in range(1, 108+1)]
    glm_ds = glm_ds.sel(puzzle=puzzle_list, stat=stat)
    glm_array = glm_ds.to_array().to_numpy()

    # get the parcellation
    parc = Parcellation(cortex=cortex, cortex_res=cortex_res, scale=scale)

    # load event information
    event_df = get_subject_events(subj_str)

    # creates lists for the RSA description
    # puzzles
    data = event_df.sort_values('LST_id')['LST_id'].to_list()
    puzzles = [str(item).zfill(3) for item in data]

    # conditions
    data = event_df.sort_values('LST_id')['LST_id_orig'].to_list()
    conds = [str(item).zfill(2) for item in data]

    # runs
    runs = event_df.sort_values('LST_id')['run'].to_list()

    # cross validation check
    
    # init the rsa toolbox list object and details
    rsd_data = []

    # loop through parcels
    for i in range(parc.n_parcels):

        # define parcel indicies, index data
        # acount for 0 indicating no parcel in parcellation
        parcel_idx = (parc.img == i+1).reshape(-1)
        data = np.squeeze(glm_array[:, :, parcel_idx, :]).T

        # create rsa toolbox object
        des = {'subject': subj_str, 'parcel': f"{str(i).zfill(3)}_{parc.parcel_labels[i]}"}
        obs_des = {'puzzles': puzzles[0:108],
                   'conds': conds[0:108],
                   'runs': runs[0:108]}
        chn_des = {'vertices': np.array(
            ['vertex_' + str(x) for x in range(data.shape[1])])}
        ds = rsd.Dataset(measurements=data,
                         descriptors=des,
                         obs_descriptors=obs_des,
                         channel_descriptors=chn_des
                         )
        # append to list
        rsd_data.append(ds)
    return rsd_data


def get_subject_events(subject):

    events_file = f"../data/fMRI/events/sub-{subject}_events.tsv"
    if isfile(events_file):
        df = pd.read_csv(events_file, delimiter="\t")
    
    # else:

    #     # load and save for next time
    #     bids_path = '/home/lukeh/hpcworking/shared/projects/LST7T/data/bids/'
    #     indexer = BIDSLayoutIndexer(validate=False, index_metadata=False)
    #     layout = BIDSLayout(bids_path, indexer=indexer)
    #     events = sorted(layout.get(task='LST', subject=subject,
    #                                extension='.tsv', return_type='filename'))
    #     assert len(events) == 3, "Number of event files is wrong"

    #     # get events from BIDS and account for n runs in the onset time
    #     df = pd.DataFrame()
    #     for run in range(len(events)):
    #         _df = pd.read_csv(events[run], delimiter='\t')
    #         _df['run'] = run+1
    #         df = pd.concat([df, _df])

    #     df.to_csv(events_file, sep="\t", index=False)
    return df
