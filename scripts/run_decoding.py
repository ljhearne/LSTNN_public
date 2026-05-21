'''
Performs the supplementary decoding analysis: tests whether task features 
can be recovered from fMRI activations
'''
from lstnn.compare_rdms import get_transformer_weights
from lstnn.dataset import get_dataset
from lstnn.generate_fMRI_rdms import load_group_data
from lstnn.parcellation import Parcellation
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from joblib import Parallel, delayed
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import nibabel as nb


# Parameters
data_dir = "projects/LSTNN/data/"
fmri_data_dir = "projects/LSTNN/data/fMRI/"

# these are derived based on head motion
subject_list = [2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15,
                17, 18, 19, 21, 23, 24, 25, 26, 27,
                28, 30, 34, 35, 36, 38, 39, 40, 41,
                42, 45, 46, 50, 51, 54, 55, 56, 61,
                64, 65]

cortex = 'Glasser'
cortex_res = None
scale = 1

# get the parcellation
parc = Parcellation(cortex=cortex, cortex_res=cortex_res, scale=scale)


def loo_cv(X, y, model, train_index, test_index):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train the model
    model.fit(X_train, y_train)

    # Predict the test sample
    return y_test[0], model.predict(X_test)[0]


# Function to find the optimal number of clusters using silhouette score
def find_optimal_clusters(data, max_k=54):
    iters = range(2, max_k+1)
    s_scores = []

    for k in iters:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        s_scores.append(silhouette_score(data, kmeans.labels_))

    optimal_k = iters[s_scores.index(max(s_scores))]
    return optimal_k


def task_condition_target():
    # Binary / Ternary / Quaternary conditions
    y = np.zeros((108))
    y[36:72] = 1
    y[72::] = 2
    return y


def visual_kmeans(LST_puzzle_ds, k=54):

    # Load the one-hot encoded visual data
    data = np.zeros((108, 80))
    for i in range(108):
        data[i, :] = LST_puzzle_ds[i][0].reshape(-1).numpy()
    # Find the optimal number of clusters
    optimal_clusters = find_optimal_clusters(
        data, k)  # You can adjust the max_k value

    # Initialize the KMeans model with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)

    # Fit the model to your data
    kmeans.fit(data)

    # Get the cluster labels for each row
    labels = kmeans.labels_

    # Print the optimal number of clusters and the cluster labels
    print(f"Optimal number of clusters: {optimal_clusters}")
    return np.array(labels)


def model_wrapper(X, y, n_jobs):
    # Initialize Leave-One-Out cross-validator
    loo = LeaveOneOut()

    # Initialize model
    model = OneVsRestClassifier(
        LogisticRegression(solver='liblinear'))

    # Perform LOO-CV in parallel
    results = Parallel(n_jobs=n_jobs)(delayed(loo_cv)(
        X, y, model, train_index, test_index) for train_index, test_index in loo.split(X))

    # Separate true and predicted values
    y_true, y_pred = zip(*results)

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy


def run_ann_decoding(target, pe_desc="2dpe", epoch=4000, n_jobs=8):
    # load test puzzles as a torch ds
    LST_puzzle_ds = get_dataset(f"{data_dir}nn/puzzle_data_original.csv")

    # get specified ANN models
    weights = get_transformer_weights(LST_puzzle_ds, pe_desc, epoch)

    # init. output
    decoding_accuracy = []

    print(target)
    # define the target
    if target == "task_condition":
        y = task_condition_target()

    elif target == "motor_response":
        # this is different for each seed.
        y_all = weights['responses']

    elif target == "visual_kmeans":
        y = visual_kmeans(LST_puzzle_ds, k=54)

    else:
        print("Target not known")

    # Loop through models architecture
    for label in ["pe", "attn", "attn_out", "mlp"]:
        print(label)
        w_data = weights[label]

        # Loop through seeds
        for s, subj in tqdm(enumerate(range(w_data.shape[0]))):

            # Perform average (no layer) decoding
            # define X
            X = np.mean(w_data[subj, :, :, :], axis=0)
            if target == "motor_response":
                y = y_all[s, :]

            # run the model
            accuracy = model_wrapper(X, y, n_jobs)

            # save
            df = pd.DataFrame()
            df["subj"] = [subj]
            df["layer"] = "average"
            df["accuracy"] = accuracy
            df["ann_desc"] = label
            df["target"] = target
            decoding_accuracy.append(df)

            # Perform decoding by layer
            for layer in range(w_data.shape[1]):

                # define X
                X = w_data[subj, layer, :, :]
                if target == "motor_response":
                    y = y_all[s, :]

                # run the model
                accuracy = model_wrapper(X, y, n_jobs)

                # save
                df = pd.DataFrame()
                df["subj"] = [subj]
                df["layer"] = layer
                df["accuracy"] = accuracy
                df["ann_desc"] = label
                df["target"] = target
                decoding_accuracy.append(df)

    results = pd.concat(decoding_accuracy)
    outname = f"../results/decoding/target-{target}_model-{pe_desc}_epoch-{epoch}.csv"
    results.to_csv(outname, index=False)
    return results


def run_fmri_decoding(target, denoise="14p", stat="t", n_jobs=4):
    decoding_accuracy = []

    # define the target
    if target == "task_condition":
        y = task_condition_target()

    elif target == "motor_response":
        # this is different for each seed.
        y = None

    elif target == "visual_kmeans":
        # load test puzzles as a torch ds
        LST_puzzle_ds = get_dataset(f"{data_dir}nn/puzzle_data_original.csv")
        y = visual_kmeans(LST_puzzle_ds, k=54)

    else:
        print("Target not known")

    for subj in subject_list:

        # load subject data
        print(subj)
        subj_data = load_group_data([subj], stat, denoise)

        # define parcel indicies, index data
        # acount for 0 indicating no parcel in parcellation
        for i in tqdm(range(parc.n_parcels)):
            parcel_idx = (parc.img == i+1).reshape(-1)
            X = subj_data[:, parcel_idx].copy()

            if target == "motor_response":
                # individualised per subject
                subj_str = str(subj).zfill(2)
                filenames = []
                for run in range(1, 4):
                    filenames.append(pd.read_csv(f"/home/lukeh/hpcworking/shared/projects/LST7T/data/bids/sub-{subj_str}/func/sub-{subj_str}_task-LST_run-{run}_events.tsv",
                                                 delimiter="\t"))
                df = pd.concat(filenames).sort_values(by="LST_id")
                y = df.motor_key.values[0:108]
                keep_trials = y != 999
                y = y[keep_trials]
                X = X[keep_trials, :]

            # Initialize Leave-One-Out cross-validator
            loo = LeaveOneOut()

            # Initialize model
            model = OneVsRestClassifier(LogisticRegression(solver='liblinear'))

            # Perform LOO-CV in parallel
            results = Parallel(n_jobs=n_jobs)(delayed(loo_cv)(
                X, y, model, train_index, test_index) for train_index, test_index in loo.split(X))

            # Separate true and predicted values
            y_true, y_pred = zip(*results)

            # Calculate accuracy
            accuracy = accuracy_score(y_true, y_pred)

            df = pd.DataFrame()
            df["subj"] = [subj]
            df["accuracy"] = accuracy
            df["region"] = parc.parcel_labels[i]
            decoding_accuracy.append(df)
    results = pd.concat(decoding_accuracy)
    outname = f"../results/decoding/target-{target}_model-fMRI_parc-glasser_denoise-{denoise}_stat-{stat}.csv"
    results.to_csv(outname, index=False)


def run_fmri_network_decoding(target, denoise="14p", stat="t", n_jobs=4):

    decoding_accuracy = []

    # define the target
    if target == "task_condition":
        y = task_condition_target()

    elif target == "motor_response":
        # this is different for each seed.
        y = None

    elif target == "visual_kmeans":
        # load test puzzles as a torch ds
        LST_puzzle_ds = get_dataset(f"{data_dir}nn/puzzle_data_original.csv")
        y = visual_kmeans(LST_puzzle_ds, k=54)

    else:
        print("Target not known")

    for subj in subject_list:

        # load subject data
        print(subj)
        subj_data = load_group_data([subj], stat, denoise)

        # define *network* indicies, index data
        network_df = pd.read_csv(parc.network_details, sep="\t").dropna()

        # combine visual 1 and two networks
        #network_df.replace(to_replace=["Visual1", "Visual2"], value="Visual", inplace=True)
        networks_of_interest = ["Visual1", "Frontoparietal", "Somatomotor"]
        for net in tqdm(networks_of_interest):

            # get glasser labels
            net_df = network_df.loc[network_df.NETWORK == net]
            glasser_labels = net_df["GLASSERLABELNAME"].to_list()

            # match glasser labels to actual (tian) parcellation used
            # and get indicies
            index = []
            img = nb.load(parc.file).get_fdata()
            for i, label in enumerate(parc.parcel_labels):
                if label in glasser_labels:
                    index.append(i+1)  # account for 0 in parcellation image

            vertex_idx = np.where(np.isin(img.reshape(-1), index))[0]

            # apply to data
            X = subj_data[:, vertex_idx].copy()

            if target == "motor_response":
                # individualised per subject
                subj_str = str(subj).zfill(2)
                filenames = []
                for run in range(1, 4):
                    filenames.append(pd.read_csv(f"/home/lukeh/hpcworking/shared/projects/LST7T/data/bids/sub-{subj_str}/func/sub-{subj_str}_task-LST_run-{run}_events.tsv",
                                                 delimiter="\t"))
                df = pd.concat(filenames).sort_values(by="LST_id")
                y = df.motor_key.values[0:108]
                keep_trials = y != 999
                y = y[keep_trials]
                X = X[keep_trials, :]

            # Initialize Leave-One-Out cross-validator
            loo = LeaveOneOut()

            # Initialize model
            model = OneVsRestClassifier(LogisticRegression(solver='liblinear'))

            # Perform LOO-CV in parallel
            results = Parallel(n_jobs=n_jobs)(delayed(loo_cv)(
                X, y, model, train_index, test_index) for train_index, test_index in loo.split(X))

            # Separate true and predicted values
            y_true, y_pred = zip(*results)

            # Calculate accuracy
            accuracy = accuracy_score(y_true, y_pred)

            df = pd.DataFrame()
            df["subj"] = [subj]
            df["accuracy"] = accuracy
            df["network"] = net
            decoding_accuracy.append(df)
    results = pd.concat(decoding_accuracy)
    outname = f"../results/decoding/target-{target}_model-fMRI_parc-glasserNetwork_denoise-{denoise}_stat-{stat}.csv"
    results.to_csv(outname, index=False)


if __name__ == '__main__':

    n_jobs = 8
    run_ann_decoding("task_condition", n_jobs=n_jobs)
    run_ann_decoding("motor_response", n_jobs=n_jobs)
    run_ann_decoding("visual_kmeans", n_jobs=n_jobs)

    # run_fmri_decoding("task_condition", denoise="14p", stat="t", n_jobs=n_jobs)
    # run_fmri_decoding("motor_response", denoise="14p", stat="t", n_jobs=n_jobs)
    # run_fmri_decoding("visual_kmeans", denoise="14p", stat="t", n_jobs=n_jobs)

    # run_fmri_network_decoding(
    #     "task_condition", denoise="14p", stat="t", n_jobs=n_jobs)
    # run_fmri_network_decoding(
    #     "motor_response", denoise="14p", stat="t", n_jobs=n_jobs)
    # run_fmri_network_decoding(
    #     "visual_kmeans", denoise="14p", stat="t", n_jobs=n_jobs)
