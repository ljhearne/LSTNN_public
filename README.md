# LSTNN_public
Public code for the LSTNN project "Aligning transformer circuit mechanisms to neural representations in relational reasoning" published in TBA.

## Notes:
- "lstnn" contains scripts to perform the fMRI analyses and ANN models. 
    - Follow https://goodresearch.dev/index.html to install the "lstnn" package, or copy functions as needed. 
    - Note these more complex functions can't be reproduced here as the input data are too large (i.e., neuroimaging files, or pytorch save states). For example, the RSA can't be performed as the input CIFTI files are not on github.
- "processed_data" contains processed data that is small enough to upload to github so results can be plotted. So, no actual CIFTIs or similar.
-  "scripts" contains high-level scripts to perform the analyses.
    - Again some of these require data too big to put on github (e.g., run_decoding relies on neuroimaging data)
    - The following notebooks draw upon "processed_data" and reproduce the results presented in the manuscript:
        - Analysis 1:
        - Analysis 2:
        - Analysis 3:
        - Analysis 4:
        - Analysis 5:

