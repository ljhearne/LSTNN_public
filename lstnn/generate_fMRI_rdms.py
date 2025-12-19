import numpy as np
import xarray as xr
import numpy as np
import pandas as pd
from tqdm import tqdm
from lstnn.parcellation import Parcellation
import rsatoolbox
import argparse

# Create parser for options
parser = argparse.ArgumentParser(
    description='''Generate fMRI RDMS''')

# These parameters must be passed to the function
parser.add_argument('--rdm_method',
                    type=str,
                    default=None)

parser.add_argument('--atlas',
                    type=str,
                    default=None)

parser.add_argument('--out',
                    type=str,
                    default=None)

parser.add_argument('--overwrite',
                    action='store_true')

# Global variables that I don't plan on changing in the pipeline
data_in = "/home/lukeh/projects/LSTNN/data/fMRI/"
denoise = "14p"
stat = "t"
subject_list = [2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15,
                17, 18, 19, 21, 23, 24, 25, 26, 27,
                28, 30, 34, 35, 36, 38, 39, 40, 41,
                42, 45, 46, 50, 51, 54, 55, 56, 61,
                64, 65]


def load_group_data(subject_list, stat, denoise):
    '''
    Loads vertex data for the given subjects, stat (z, t, b)
    and denoising pipeline and returns a large xarray
    '''

    all_glm_data = np.zeros((108*len(subject_list), 91282))
    index = 0
    for subj in subject_list:

        # load LS GLM file with 144 trials
        subj_str = str(subj).zfill(2)
        glm_in = f"{data_in}fmriprep_denoised_glm_ls1/sub-{subj_str}/"
        glm_in += f"sub-{subj_str}_task-LST_space-fsLR_"
        glm_in += f"denoise-{denoise}_model-144_glm.nc"

        # combine into an xarray dataset
        glm_ds = xr.open_mfdataset(glm_in, concat_dim='subject',
                                   combine='nested')

        # reorder all the puzzles from Binary -> Ternary -> Quaternary -> Null
        # and select the specified stat
        puzzle_list = ['LST' + str(i) for i in range(1, 108+1)]
        glm_ds = glm_ds.sel(puzzle=puzzle_list, stat=stat)
        glm_array = glm_ds.to_array().to_numpy()

        # concat
        all_glm_data[index:index+108, :] = np.squeeze(glm_array).T
        index = index+108
    return all_glm_data


def get_group_roi_ds(all_glm_data, subject_list, roi, parc):
    '''
    Generates a single rsatoolbox dataset with all subjects and trials
    for a single region of interest for the given parcellation file
    '''

    index = 0
    all_puzzles = np.zeros((108*len(subject_list)))
    all_subjects = np.zeros((108*len(subject_list)))

    for i, subj in enumerate(subject_list):
        subj_str = str(subj).zfill(2)
        all_subjects[index:index+108] = i

        # load event information
        event_df = pd.read_csv(f"{data_in}events/sub-{subj_str}_events.tsv",
                               delimiter="\t")

        # creates lists for the RSA description
        # puzzles
        data = event_df.sort_values('LST_id')['LST_id'].to_list()
        puzzles = [str(item).zfill(3) for item in data]
        all_puzzles[index:index+108] = puzzles[0:108]

        # index
        index = index+108

    # loop through parcels
    i = np.where(parc.parcel_labels == roi)[0][0]

    # define parcel indicies, index data
    # acount for 0 indicating no parcel in parcellation
    parcel_idx = (parc.img == i+1).reshape(-1)
    data = all_glm_data[:, parcel_idx]

    # create rsa toolbox object
    des = {'region': f"{str(i).zfill(3)}_{parc.parcel_labels[i]}"}
    obs_des = {'puzzles': all_puzzles,
               'subjects': all_subjects}

    chn_des = {'vertices': np.array(
        ['vertex_' + str(x) for x in range(data.shape[1])])}
    ds = rsatoolbox.data.Dataset(measurements=data,
                                 descriptors=des,
                                 obs_descriptors=obs_des,
                                 channel_descriptors=chn_des
                                 )
    return ds


def generate_group_rdms(rdm_method, out, atlas, overwrite):

    if atlas == "Glasser":
        cortex = 'Glasser'
        cortex_res = None
        scale = 1
    elif atlas == "Schaefer":
        cortex = 'Schaefer'
        cortex_res = 400
        scale = 1

    # get the parcellation
    parc = Parcellation(cortex=cortex, cortex_res=cortex_res, scale=scale)

    # get group data
    all_glm_data = load_group_data(subject_list, stat, denoise)

    # generate all the fMRI RDMS
    for roi in tqdm(parc.parcel_labels):
        roi_ds = get_group_roi_ds(all_glm_data, subject_list, roi, parc)

        # cross validated rdm
        # if euclidean we don't cross-validate
        if rdm_method == "euclidean":
            rdm_cv = rsatoolbox.rdm.calc_rdm(roi_ds, method=rdm_method,
                                             descriptor='puzzles')
        else:
            rdm_cv = rsatoolbox.rdm.calc_rdm(roi_ds, method=rdm_method,
                                             descriptor='puzzles',
                                             cv_descriptor='subjects')
        # out
        rdm_cv.save(f"{out}_roi-{roi}.h5", overwrite=overwrite)


if __name__ == '__main__':
    # Read in user-specified parameters
    args = parser.parse_args()

    # run
    generate_group_rdms(args.rdm_method, args.out,
                        args.atlas, args.overwrite)
