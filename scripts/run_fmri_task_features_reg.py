# %%
from lstnn.dataset import get_dataset
import rsatoolbox
import numpy as np
import pandas as pd
from lstnn.compare_rdms import get_transformer_weights, get_transformer_rdms
from lstnn.task_features import get_experimentor_rdms
from sklearn.linear_model import LinearRegression
from lstnn.parcellation import Parcellation
from tqdm import tqdm
from joblib import Parallel, delayed

# parameters
data_dir = "/home/lukeh/projects/LSTNN/data/"
rdm_method_ann = "euclidean"
pe_desc = "2dpe"
epoch = 4000
atlas = "Glasser"
n_perms = 10000
group = "group"
rdm_method_fmri = "crossnobis"

out = (
    f"/home/lukeh/projects/LSTNN/results/model_comparison_task_features_reg/{group}_"
    f"atlas-{atlas}/fmethod-{rdm_method_fmri}_"
    f"cmethod-regression_"
    f"nperms-{n_perms}_roi-"
)

def get_X(reg_model, baseline_data, task_models):
    if reg_model == "baseline":
        X = baseline_data.reshape(1, -1)

    elif reg_model == "stimulus":
        X = np.vstack((baseline_data,
                    task_models[0].rdm
                    ))
        
    elif reg_model == "complexity":
        X = np.vstack((baseline_data, 
                    task_models[1].rdm
                    ))
        
    elif reg_model== "all":
        X = np.vstack((baseline_data, 
                    task_models[0].rdm,
                    task_models[1].rdm,
                    task_models[2].rdm
                    ))
    return X


def run_regression(y, task_models, ann_models, roi, permute):
    res = []
    for model_index, model_label in zip([4,9], ["pe_avg", "attn_out_avg"]):
        for reg_model in ["baseline", "stimulus", "complexity", "all"]:
            # ann_models[4].name = "pe_avg"
            # ann_models[9].name = 'attn_out_avg'
            if permute:
                # shuffle the ANN model and keep the task models the same.
                model = ann_models[model_index].rdm_obj
                p = np.random.permutation(model.n_cond)
                t = rsatoolbox.rdm.rdms.permute_rdms(model, p=p)
                X = get_X(reg_model, 
                            t.dissimilarities,
                            task_models)

            else:
                X = get_X(reg_model, 
                        ann_models[model_index].rdm,
                        task_models)

            reg = LinearRegression().fit(X.T, y.T)
            df = pd.DataFrame()
            df['permutation'] = [permute]
            df["reg_model"] = reg_model
            df['stat'] = reg.coef_[0][0]
            df['compare_method'] = "regression"
            df['model'] = model_label
            df['parcel'] = roi
            res.append(df)
    return pd.concat(res)
    

# load test puzzles as a torch ds
LST_puzzle_ds = get_dataset(f"{data_dir}nn/puzzle_data_original.csv")

# get specified ANN models
weights = get_transformer_weights(LST_puzzle_ds, pe_desc, epoch)
rdms = get_transformer_rdms(weights, rdm_method_ann)

# combine into a list of rsatoolbox models
ann_models = []
for rdm in rdms:
    label = rdm.rdm_descriptors['name'][0]
    m = rsatoolbox.model.ModelFixed(label, rdm)
    ann_models.append(m)

task_rdms = get_experimentor_rdms(LST_puzzle_ds)

# Combine into a list of rsatoolbox models
task_models = []
for rdm in task_rdms:
    label = rdm.rdm_descriptors['name'][0]
    m = rsatoolbox.model.ModelFixed(label, rdm)
    task_models.append(m)

# load fmri empirical rdms
if atlas == "Glasser":
    cortex = 'Glasser'
    cortex_res = None
    scale = 1
elif atlas == "Schaefer":
    cortex = 'Schaefer'
    cortex_res = 400
    scale = 1

parc = Parcellation(cortex=cortex, cortex_res=cortex_res, scale=scale)


for roi in parc.parcel_labels:
    print(roi)
    # get fmri rdm
    rdm_file = f"{data_dir}fMRI/rdms/{group}_atlas-{atlas}/"
    rdm_file += f"method_{rdm_method_fmri}_roi-{roi}.h5"
    fmri_rdm = rsatoolbox.rdm.load_rdm(rdm_file)

    # init reg model
    y = fmri_rdm.dissimilarities


    # real data
    results = run_regression(y, task_models, ann_models, roi, permute=False)

    # permutations
    perm_results = Parallel(n_jobs=12, backend="loky")(
        delayed(run_regression)(y, task_models, ann_models, roi, permute=True) for i in tqdm(range(n_perms))
    )

    roidf = pd.concat([results, pd.concat(perm_results)])

    # save interim results out
    roidf.to_csv(f"{out}{roi}.csv", index=False)

# %%
