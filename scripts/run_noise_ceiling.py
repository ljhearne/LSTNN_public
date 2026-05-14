import os
from scipy.stats import pearsonr
from lstnn.generate_fMRI_rdms import load_group_data, get_group_roi_ds
from lstnn.parcellation import Parcellation
from tqdm import tqdm
import numpy as np
import pandas as pd
import rsatoolbox
from joblib import Parallel, delayed

# Global variables that I don't plan on changing in the pipeline
data_in = "/home/lukeh/projects/LSTNN/data/fMRI/"
denoise = "14p"
stat = "t"
subject_list = [2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15,
                17, 18, 19, 21, 23, 24, 25, 26, 27,
                28, 30, 34, 35, 36, 38, 39, 40, 41,
                42, 45, 46, 50, 51, 54, 55, 56, 61,
                64, 65]
atlas = "Glasser"
rdm_method = "crossnobis"
n_perms = 1000
group = "group"
out = (
    f"/home/lukeh/projects/LSTNN/results/noise_ceilings/{group}_"
    f"atlas-{atlas}/fmethod-{rdm_method}_"
    f"cmethod-spearman_"
    f"nperms-{n_perms}_roi-"
)
overwrite = True

def random_split_half(subjects, seed=None):
    """
    Randomly splits a list of subjects into two disjoint groups of equal size.
    
    Parameters
    ----------
    subjects : list
        List of subject identifiers (length must be even; typically 40).
    seed : int or None
        Optional random seed for reproducibility.
    
    Returns
    -------
    group1 : list
    group2 : list
    """
    if seed is not None:
        np.random.seed(seed)

    subjects = np.array(subjects)
    n = len(subjects)
    if n % 2 != 0:
        raise ValueError("Number of subjects must be even for split-half.")

    # random permutation
    perm = np.random.permutation(n)

    # split into equal halves
    group1 = subjects[perm[:n//2]].tolist()
    group2 = subjects[perm[n//2:]].tolist()

    return group1, group2


def perm_worker(subject_list, all_glm_data, subject_index, roi, parc, rdm_method):
    # split the subject list
    g1, g2 = random_split_half(subject_list)

    # get group data
    # subset glm data based on the above
    mask = np.isin(subject_index, g1).astype(int)
    glm_data_a = all_glm_data[mask==1, :]
    mask = np.isin(subject_index, g2).astype(int)
    glm_data_b = all_glm_data[mask==1, :]

    # compute RDMs for this ROI and get rho
    roi_ds_a = get_group_roi_ds(glm_data_a, g1, roi, parc)
    rdm_cv_a = rsatoolbox.rdm.calc_rdm(
        roi_ds_a, method=rdm_method,
        descriptor='puzzles', cv_descriptor='subjects'
    )

    roi_ds_b = get_group_roi_ds(glm_data_b, g2, roi, parc)
    rdm_cv_b = rsatoolbox.rdm.calc_rdm(
        roi_ds_b, method=rdm_method,
        descriptor='puzzles', cv_descriptor='subjects'
    )

    r, _ = pearsonr(
        np.ravel(rdm_cv_a.dissimilarities),
        np.ravel(rdm_cv_b.dissimilarities)
    )
    return r


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

# get ALL data
all_glm_data = load_group_data(subject_list, stat, denoise)
print("GLM data loaded...")

# create a vector that indicates which subject each row of data belongs to
subject_index = np.zeros((108*len(subject_list)))

index = 0
for i, subj in enumerate(subject_list):
    subject_index[index:index+108] = subj
    index = index+108

# loop through ROIs
roi_list = list(parc.parcel_labels)
# roi_list = ['R_V1_ROI', 'R_V2_ROI', 'R_V3_ROI', 'R_8Av_ROI','R_a47r_ROI',
#  'R_s6-8_ROI', 'R_TE1p_ROI', 'R_PGs_ROI', 'R_p47r_ROI', 'L_V1_ROI', 'L_V2_ROI',
#  'L_V3_ROI', 'L_a47r_ROI', 'L_PGs_ROI', 'L_p47r_ROI']

for roi in roi_list:
    res = []
    print("Current ROI:", roi)

    # if file exists, skip
    if os.path.exists(f"{out}{roi}.csv") and not overwrite:
        print(f"File exists, skipping: {out}{roi}.csv")
    else:

        res = Parallel(n_jobs=12, backend="loky")(
            delayed(perm_worker)(subject_list, all_glm_data, subject_index, roi, parc, rdm_method)
            for i in tqdm(range(n_perms))
        )

        roidf = pd.DataFrame()
        roidf["rho"] = res
        roidf["roi"] = roi
        roidf.to_csv(f"{out}{roi}.csv", index=False)
        print(f"Saved: {out}{roi}.csv")