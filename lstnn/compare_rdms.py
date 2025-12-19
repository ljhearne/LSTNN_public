import numpy as np
import pandas as pd
import rsatoolbox
from lstnn.dataset import get_dataset
import lstnn.transformer_main as transformer_main
from lstnn.parcellation import Parcellation
import torch
from copy import deepcopy
from joblib import Parallel, delayed
from tqdm import tqdm
from sys import exit
from os.path import isfile
import argparse

# Create parser for options
parser = argparse.ArgumentParser(
    description='''Run RDM comparison''')

# These parameters must be passed to the function
parser.add_argument('--pe_desc',
                    type=str,
                    default=None)

parser.add_argument('--rdm_method_fmri',
                    type=str,
                    default=None)

parser.add_argument('--rdm_method_ann',
                    type=str,
                    default=None)

parser.add_argument('--compare_method',
                    type=str,
                    default=None)

parser.add_argument('--out',
                    type=str,
                    default=None)

parser.add_argument('--group',
                    type=str,
                    default=None)

parser.add_argument('--epoch',
                    type=int,
                    default=None)

parser.add_argument('--n_perms',
                    type=int,
                    default=None)

parser.add_argument('--atlas',
                    type=str,
                    default=None)

parser.add_argument('--overwrite',
                    action='store_true')

# global parameters unlikely to change:
n_jobs = 24

# ANN
data_dir = "/home/lukeh/projects/LSTNN/data/"
data_dir_ldrive = "/home/lukeh/LabData/Lab_LucaC/Luke/projects/LSTNN/data/"
seeds = [2235, 6312, 6068, 9742, 8880, 2197, 669,
         6256, 3309, 2541, 8643, 7785, 195, 6914, 29]
model_label = 'BERT'
n_layer = 4
dropout = 0.0
wdecay = 0.0
attnhead = 1
hidden_size = 160
curriculum = 'All'
learning_rate = 0.0001
device = 'cpu'
training_acc_cutoff = 0.0
cutoff_length = 0

# fMRI
cortex = 'Glasser'
cortex_res = None
scale = 1


def retrieve_model_weights(model, batch):
    # This function extracts weights and intermediate
    # outputs from a transformer model given a specific input batch.

    # Initialize lists to store outputs from each transformer block
    attn_weights = []   # List to store attention weight matrices
    attn_out = []       # List to store outputs after attention layer
    mlp_weights = []    # List to store outputs from the MLP layer
    pe_embed = []       # List to store outputs after positional encoding

    # Compute initial word embeddings from the input batch
    embedding = model.w_embed(batch)

    # Iterate over each transformer block in the model
    for block in model.blocks:

        # Apply positional encoding and store result
        embedding = block.pe(embedding)
        pe_embed.append(block.pe(embedding))

        # Apply dropout to the embeddings
        embedding = block.dropout_embed(embedding)

        # Compute self-attention outputs and attention weights
        attn_outputs, attn_weights_layer = block.selfattention(
            embedding, embedding, embedding, need_weights=True
        )

        # Apply first residual connection and layer normalization
        attn_outputs = block.layernorm0(attn_outputs + embedding)

        # Pass through MLP layer
        transformer_out = block.mlp(attn_outputs)

        # Apply second residual connection and layer normalization
        embedding = block.layernorm1(transformer_out + attn_outputs)

        # Store collected outputs for analysis or visualization
        attn_weights.append(attn_weights_layer)
        mlp_weights.append(transformer_out)
        attn_out.append(attn_outputs)

    return attn_weights, mlp_weights, attn_out, pe_embed


def create_ds_from_ann(weights, label):
    # assumes a very specific shape for weights
    n_seeds = weights.shape[0]
    assert weights.shape[1] == 108

    data = np.swapaxes(weights, 0, 1)
    data = np.reshape(data, (-1, data.shape[2]), order='F')
    des = {'name': label}

    # generate information for puzzles, conditions and seeds
    puzzles = np.tile(np.arange(1, 109), n_seeds)

    seed_vec = []
    for seed in seeds:
        seed_vec.append([seed] * 108)
    seed_vec = np.array(seed_vec).reshape(-1)

    obs_des = {'puzzles': puzzles,
               'seeds': seed_vec}

    ds = rsatoolbox.data.Dataset(measurements=data,
                                 descriptors=des,
                                 obs_descriptors=obs_des
                                 )
    return ds


def get_transformer_weights(LST_puzzle_ds, pe_desc, epoch):
    """
    Load trained transformer models with different seeds, extract and return
    attention weights, attention outputs, MLP outputs, and response predictions
    for a given dataset and positional encoding description.
    """

    # Determine positional encoding type based on description string
    if "learn" in pe_desc:
        petype = "learn"
    elif "2dpe" in pe_desc:
        petype = "absolute2d"
    elif "1dpe" in pe_desc:
        petype = "absolute"
    elif "rndpe" in pe_desc:
        petype = "rndpe"

    # Build model path string using hyperparameters
    modelname = f"model-{model_label}_" \
                f"pe-{pe_desc}_" \
                f"nl-{n_layer}_" \
                f"do-{dropout}_" \
                f"wd-{wdecay}_" \
                f"at-{attnhead}_" \
                f"hs-{hidden_size}_" \
                f"curr-{curriculum}_" \
                f"lr-{learning_rate}_" \
                f"co-{training_acc_cutoff}_" \
                f"col-{cutoff_length}/"

    # Prepare data loader for evaluation
    dataloader = torch.utils.data.DataLoader(LST_puzzle_ds,
                                             batch_size=108,
                                             shuffle=False)

    # Initialize containers to hold model outputs for all seeds
    attn_weights_all = []  # Stores attention weights
    mlp_weights_all = []   # Stores MLP outputs
    attn_out_all = []      # Stores outputs of attention layer
    pe_embed_all = []      # Stores positional encoding outputs
    response_all = []      # Stores final predictions

    # Loop through each seed to evaluate models with different initializations
    for seed in seeds:
        torch.manual_seed(seed)  # Set seed for reproducibility

        # Instantiate and load model with saved parameters
        model = transformer_main.Transformer(
            nblocks=n_layer,
            nhead=attnhead,
            dropout=dropout,
            embedding_dim=hidden_size,
            positional_encoding=petype
        )
        model = model.to(device=torch.device(device))

        model_file = f"{data_dir_ldrive}ito_models/{modelname}s-{seed}_e-{epoch}.pt"
        model.load_state_dict(torch.load(
            model_file, map_location=torch.device(device)))

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                test_features = batch[0]  # Get inputs from batch
                test_features = torch.flatten(
                    test_features, start_dim=1, end_dim=2)  # Flatten inputs
                test_features = test_features.to(device)

                # Retrieve internal transformer weights and outputs
                attn_weights, mlp_weights, attn_out, pe_embed = retrieve_model_weights(
                    model, test_features)

                # Save outputs for this model/seed
                attn_weights_all.append(torch.stack(attn_weights))
                mlp_weights_all.append(torch.stack(mlp_weights))
                attn_out_all.append(torch.stack(attn_out))
                pe_embed_all.append(torch.stack(pe_embed))

                # Store the model’s predicted response
                response_all.append(torch.argmax(model(test_features), dim=1))

    # Stack and reshape collected outputs
    attn_weight_avg2d = torch.stack(attn_weights_all)
    attn_weights = attn_weight_avg2d.reshape(len(seeds), n_layer, 108, -1)

    attn_out_avg2d = torch.stack(attn_out_all)
    attn_out_weights = attn_out_avg2d.reshape(len(seeds), n_layer, 108, -1)

    pe_avg2d = torch.stack(pe_embed_all)
    pe_weights = pe_avg2d.reshape(len(seeds), n_layer, 108, -1)

    mlp_weight_avg2d = torch.stack(mlp_weights_all)
    mlp_weights = mlp_weight_avg2d.reshape(len(seeds), n_layer, 108, -1)

    response_avg2d = torch.stack(response_all)
    responses = response_avg2d.reshape(len(seeds), 108)

    # Return results as NumPy arrays for further analysis
    return {
        "pe": pe_weights.numpy(),
        "attn": attn_weights.numpy(),
        "attn_out": attn_out_weights.numpy(),
        "mlp": mlp_weights.numpy(),
        "responses": responses.numpy()
    }


def get_transformer_rdms(weights, rdm_method):
    ds_list = []
    for ann_label in ["pe", "attn_out", "mlp"]:
        for layer in range(n_layer):
            label = f"{ann_label}_layer{layer}"
            ds = create_ds_from_ann(weights[ann_label][:, layer, :, :], label)
            ds_list.append(ds)

        # averages
        label = f"{ann_label}_avg"
        ds = create_ds_from_ann(np.mean(weights[ann_label], axis=1), label)
        ds_list.append(ds)

    rdms = rsatoolbox.rdm.calc_rdm(ds_list, method=rdm_method,
                                   descriptor='puzzles',
                                   cv_descriptor='seeds')

    return rdms


def compare_rdm(fmri_rdms, models, compare_method, permute=False, it=0):
    results_df = pd.DataFrame()
    for roi in fmri_rdms:

        # deepcopy the original models
        models_i = deepcopy(models)

        # if permute, permute the models
        if permute:
            for i, m in enumerate(models_i):
                rdm = m.rdm_obj
                p = np.random.permutation(rdm.n_cond)
                t = rsatoolbox.rdm.rdms.permute_rdms(rdm, p=p)
                models_i[i].rdm_obj = t

        # model inference
        results = rsatoolbox.inference.eval_fixed(models_i,
                                                  fmri_rdms[roi],
                                                  method=compare_method)

        # collate results
        model_names = []
        for m in models_i:
            model_names.append(m.name)

        df = pd.DataFrame()
        df['stat'] = results.get_means()
        df['permutation'] = permute
        df['compare_method'] = compare_method
        df['model'] = model_names
        df['parcel'] = roi
        df['it'] = it
        results_df = pd.concat([results_df, df])
    return results_df


def compare_models_to_fmri(pe_desc, rdm_method_fmri,
                           rdm_method_ann, compare_method,
                           epoch, n_perms, group,
                           out, atlas, overwrite):

    if overwrite is False:
        # check if the output exists
        if isfile(out):
            print("Output exists, stopping code.")
            exit(0)

    # load test puzzles as a torch ds
    LST_puzzle_ds = get_dataset(f"{data_dir}nn/puzzle_data_original.csv")

    # get specified ANN models
    weights = get_transformer_weights(LST_puzzle_ds, pe_desc, epoch)
    rdms = get_transformer_rdms(weights, rdm_method_ann)

    # combine into a list of rsatoolbox models
    models = []
    for rdm in rdms:
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

    return None


if __name__ == '__main__':
    # Read in user-specified parameters
    args = parser.parse_args()

    compare_models_to_fmri(args.pe_desc,
                           args.rdm_method_fmri,
                           args.rdm_method_ann,
                           args.compare_method,
                           args.epoch,
                           args.n_perms,
                           args.group,
                           args.out,
                           args.atlas,
                           args.overwrite)
