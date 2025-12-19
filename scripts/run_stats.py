from sys import exit
from os.path import isfile
import argparse
import pandas as pd
from tqdm import tqdm
from pingouin import multicomp
import numpy as np
import argparse
from lstnn.parcellation import Parcellation

# Create parser for options
parser = argparse.ArgumentParser(
    description='''Run RDM comparison''')

# These parameters must be passed to the function
parser.add_argument('--pe_desc',
                    type=str,
                    default=None)

parser.add_argument('--group',
                    type=str,
                    default=None)

parser.add_argument('--rdm_method_fmri',
                    type=str,
                    default=None)

parser.add_argument('--rdm_method_exp',
                    type=str,
                    default=None)

parser.add_argument('--rdm_method_ann',
                    type=str,
                    default=None)

parser.add_argument('--compare_method',
                    type=str,
                    default=None)

parser.add_argument('--atlas',
                    type=str,
                    default=None)

parser.add_argument('--out',
                    type=str,
                    default=None)

parser.add_argument('--epoch',
                    type=int,
                    default=None)

parser.add_argument('--n_perms',
                    type=int,
                    default=None)

parser.add_argument('--overwrite',
                    action='store_true')


def run_statistics(pe_desc, group,
                   rdm_method_fmri, rdm_method_ann,
                   compare_method, epoch,
                   n_perms, out, atlas, overwrite):

    if overwrite is False:
        # check if the output exists
        if isfile(out):
            print("Output exists, stopping code.")
            exit(0)
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

    in_file = f"/home/lukeh/projects/LSTNN/results/model_comparison/{group}_"
    in_file += f"atlas-{atlas}/pe-{pe_desc}_"
    in_file += f"fmethod-{rdm_method_fmri}_"
    in_file += f"amethod-{rdm_method_ann}_cmethod-{compare_method}_"
    in_file += f"epoch-{epoch}_nperms-{n_perms}.csv"
    df = pd.read_csv(in_file)

    # add detail to df
    df['pe'] = pe_desc
    df["fmethod"] = rdm_method_fmri
    df['cmethod'] = compare_method
    df['amethod'] = rdm_method_ann
    df['epoch'] = epoch

    # create a df with just empirical results
    keep_df = df.loc[(df.permutation == False)]
    keep_df["p_FDR"] = np.nan
    keep_df["percentile"] = np.nan

    # calculate p-values
    for model in df.model.unique():
        print(model)
        _df = df.loc[df.model == model]
        _df_real = _df.loc[(_df.permutation == False)]
        _df_null = _df.loc[(_df.permutation == True)]

        data = []
        percentiles = []
        for parcel in tqdm(_df.parcel.unique()):
            value = _df_real.loc[(_df_real.parcel == parcel), "stat"].values[0]
            null_distribution = _df_null.loc[(
                _df_null.parcel == parcel), "stat"].values
            percentile = (null_distribution < value).mean()
            percentiles.append(percentile)
            #p_val = (0.5 - (abs(percentile - 0.5))) # Express similar to a p value regardless of direction
            p = 1 - percentile  # only investigate one direction
            data.append(p)
        # Don't analyse subcortical regions
        reject, p_corrected = multicomp(np.array(data[16::]), method="fdr_by")
        p_fdr = np.ones((len(data)))
        p_fdr[16::] = p_corrected
        keep_df.loc[(keep_df.model == model), "p_FDR"] = p_fdr.copy()
        keep_df.loc[(keep_df.model == model), "percentile"] = np.array(percentiles).copy()

    # # Assign network labels
    # keep_df["network"] = np.nan
    # network_df = pd.read_csv(
    #     '/home/lukeh/projects/RelationalRepresentations/data/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey.txt', delimiter='\t')

    # # Match the network labels to roi indices
    # for i, parcel in enumerate(keep_df.parcel.unique()):
    #     label = network_df[network_df.GLASSERLABELNAME ==
    #                        parcel].NETWORK.values
    #     if label.size == 1:
    #         label = label[0]
    #     elif label.size == 0:
    #         label = 'subcortex'
    print("Applying network labels...")
    keep_df["network"] = parc.apply_networks(keep_df["parcel"].to_list())
    keep_df.to_csv(out, index=False)
    return None


if __name__ == '__main__':
    # Read in user-specified parameters
    args = parser.parse_args()

    run_statistics(args.pe_desc, args.group,
                   args.rdm_method_fmri, args.rdm_method_ann,
                   args.compare_method, args.epoch,
                   args.n_perms, args.out, args.atlas, args.overwrite)
#             out=/home/lukeh/projects/LSTNN/results/model_comparison/${group}_atlas-${atlas}/pe-${pe_desc}
#             out=${out}_fmethod-${fmethod}_emethod-${emethod}_amethod-${amethod}_cmethod-${cmethod}
#             out=${out}_epoch-${epoch}_nperms-${n_perms}_stats.csv
#  python ../../run_stats.py  --pe_desc 2dpe --group group --rdm_method_fmri crossnobis --rdm_method_ann euclidean --compare_method corr --epoch 4000 --n_perms 10000 --out ${out}  --atlas ${atlas} #--overwrite
