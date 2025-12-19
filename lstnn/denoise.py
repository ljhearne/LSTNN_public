"""
fMRI denoising using nilearn
Uses general reccomendations from
[1] Parkes et al., 2018
An evaluation of the efficacy, reliability, and sensitivity of motion
correction strategies for resting-state functional MRI
[2] Ciric et al., 2017,
Benchmarking of participant-level confound regression strategies for the
control of motion artifact in studies of functional connectivity
[3] Lindquist et al (2018).
Modular preprocessing pipelines can reintroduce artifacts into fMRI data. bioRxiv, 407676.
In short:
    - Use GSR
    - Many regressors + derivatives and quadratics tends to perform best
        - 36P
            - 6 motion estimates
            - 2 physiological time series
            - GSR
            - All derivatives and quadratics
    - Report remaining data after volume censoring (less than 4 minutes = very bad)
    - Use a single model with filtering (this is implemented in Nilearn)
"""
import pandas as pd
import json
from nilearn.signal import clean
import numpy as np
import nibabel as nb
import argparse

# Create parser for options
parser = argparse.ArgumentParser(
    description='''Run nilearn based BOLD denoising''')

# These parameters must be passed to the function
parser.add_argument('--input',
                    type=str,
                    default=None,
                    help='''input bold data''')

parser.add_argument('--confounds',
                    type=str,
                    default=None,
                    help='''input fmriprepconfounds file''')

parser.add_argument('--strat',
                    type=str,
                    default=None,
                    help='''denoise strategy located in denoise_config.json''')

parser.add_argument('--bold_json',
                    type=str,
                    default=None,
                    help='''BIDS json file to gather TR''')

parser.add_argument('--output',
                    type=str,
                    default=None,
                    help='''output file''')


def get_bold_input(input):
    if input.endswith('.txt'):
        bold = np.loadtxt(input, delimiter=',')

    elif input.endswith('.nii'):
        bold = nb.load(input).get_fdata()

    return bold


def save_bold_output(output, bold_clean, input):
    if output.endswith('.txt'):
        np.savetxt(output, bold_clean, delimiter=',')

    elif output.endswith('.nii'):
        img = nb.load(input)
        out_img = nb.Cifti2Image(bold_clean, header=img.header, nifti_header=img.nifti_header)
        nb.save(out_img, output)


def _derivative(data):
    out = np.zeros(data.shape)
    out[0] = np.nan
    out[1:] = data[1:] - data[:-1]
    return out

def get_missing_derivatives(conf_df, conf_labels):
    for label in conf_labels:
        if label not in conf_df.columns:
            s_label = label.split('_')
            orig_label = '_'.join(s_label[0:-1])
            operation = s_label[-1]
            assert orig_label in conf_df.columns, "derivative not found in df"
            assert operation in ['derivative1', 'power2', 'derivative1_power2'], "non derivative operation unknown"

            # do the operation
            data = conf_df[orig_label].values
            if operation == 'derivative1':
                out = _derivative(data)
            elif operation == 'power2':
                out = data*data
            elif operation == 'derivative1_power2':
                out = _derivative(data)*_derivative(data)
            
            conf_df[label] = out
    return conf_df

def denoise(input, confounds, strat, bold_json, output):
    
    # load confound .tsv
    conf_df = pd.read_csv(confounds, delimiter="\t")

    # load denoise dict (assumed to be in same location)
    denoise_strat = json.load(open('../src/denoise_config.json',))[strat]

    # get tr information
    tr = json.load(open(bold_json,))['RepetitionTime']

    # organise confound variables in line with fmriprep's
    # .tsv file
    conf_labels = []
    for conf in denoise_strat["confound_variables"]:
        conf_labels.append(conf["name"])
        if conf["derivative"]:
            conf_labels.append(conf["name"] + "_derivative1")
        if conf["quadratic"]:
            conf_labels.append(conf["name"] + "_power2")
        if conf["derivative2"]:
            conf_labels.append(conf["name"] + "_derivative1_power2")

    # create regressors not included in fmriprep, if needed:
    conf_df = get_missing_derivatives(conf_df, conf_labels)

    # calculate spike regressor
    if denoise_strat["spike_regression"]["run"]:
        measure = denoise_strat["spike_regression"]["measure"]
        thresh = denoise_strat["spike_regression"]["thresh"]
        values = conf_df[measure].values
        spike_reg = (values > thresh) * 1  # (* 1 to convert to int)
        conf_df["spike_reg"] = spike_reg.copy()
        conf_labels.append("spike_reg")

    # get filter information
    if denoise_strat["filter"]["run"]:
        low_pass = denoise_strat["filter"]["low_pass"]
        high_pass = denoise_strat["filter"]["high_pass"]
    else:
        low_pass = None
        high_pass = None

    # remove nans from confound dataframe
    conf_array = conf_df[conf_labels].copy()
    conf_array .fillna(0, inplace=True)

    # print information
    print(
        "\tStarting image cleaning with " +
        str(conf_array.shape[1]) + " derivatives"
    )

    # load bold data
    # (time by space)
    bold = get_bold_input(input)

    # use nilearn to clean the image
    print("\tRunning denoising process using nilearn")
    bold_clean = clean(bold,
                        runs=None,
                        detrend=denoise_strat["detrend"],
                        standardize=denoise_strat["standardize"],
                        confounds=conf_array,
                        low_pass=low_pass,
                        high_pass=high_pass,
                        t_r=tr,
                        )

    # remove first N frames, if requested
    if denoise_strat["drop_frames"]["run"]:
        n_frames = denoise_strat["drop_frames"]["n_frames"]
        bold_clean = bold_clean[n_frames::, :]

    # save out
    save_bold_output(output, bold_clean, input)
    return bold_clean


if __name__ == '__main__':
    # Read in user-specified parameters
    args = parser.parse_args()

    # run denoise
    denoise(args.input, args.confounds, args.strat,
            args.bold_json, args.output)
