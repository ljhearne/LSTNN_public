from lstnn.compare_rdms import compare_rdm
from lstnn.parcellation import Parcellation
from tqdm import tqdm
from joblib import Parallel, delayed
import rsatoolbox
import numpy as np
import pandas as pd
from lstnn.dataset import get_dataset
from lstnn.task_features import get_experimentor_rdms

data_dir = "/home/lukeh/projects/LSTNN/data/"
atlas = "Glasser"
group = "group"
rdm_method_fmri = "crossnobis"
#compare_method = "spearman"
compare_method = "corr"
n_perms=10000
n_jobs=12

out = (
    f"/home/lukeh/projects/LSTNN/results/model_comparison_task_features/{group}_"
    f"atlas-{atlas}/fmethod-{rdm_method_fmri}_"
    f"cmethod-{compare_method}_"
    f"nperms-{n_perms}.csv"
)


# Get task RDMS
# load test puzzles as a torch ds
LST_puzzle_ds = get_dataset(f"{data_dir}nn/puzzle_data_original.csv")

task_rdms = get_experimentor_rdms(LST_puzzle_ds)

# Combine into a list of rsatoolbox models
models = []
for rdm in task_rdms:
    label = rdm.rdm_descriptors['name'][0]
    m = rsatoolbox.model.ModelFixed(label, rdm)
    models.append(m)

# load empirical rdms
if atlas == "Glasser":
    cortex = 'Glasser'
    cortex_res = None
    scale = 1
elif atlas == "Schaefer":
    cortex = 'Schaefer'
    cortex_res = 400
    scale = 1

parc = Parcellation(cortex=cortex, cortex_res=cortex_res, scale=scale)
fmri_rdms = {}
for roi in parc.parcel_labels:
    rdm_file = f"{data_dir}fMRI/rdms/{group}_atlas-{atlas}/"
    rdm_file += f"method_{rdm_method_fmri}_roi-{roi}.h5"
    fmri_rdms[roi] = rsatoolbox.rdm.load_rdm(rdm_file)

# compare the fmri data to the models
fmri_df = compare_rdm(
    fmri_rdms, models, compare_method=compare_method, permute=False)

# run permutations
res = Parallel(n_jobs=n_jobs)(
    delayed(compare_rdm)(
        fmri_rdms,
        models,
        compare_method=compare_method,
        permute=True,
        it=i
    ) for i in tqdm(range(n_perms))
)

# merge together
results_df = pd.concat(
    [fmri_df, pd.concat(res, axis=0, ignore_index=True)])

# save results out
results_df.to_csv(out, index=False)