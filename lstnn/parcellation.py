'''
A parcellation helper for Tian's parcellations


'''
import numpy as np
import nibabel as nb
import pandas as pd


class Parcellation:
    def __init__(self, cortex='Glasser', cortex_res=None, scale=1, volumetric=False,
                 path='/home/lukeh/LabData/Lab_LucaC/Luke/Backups/hpc_backups/parcellations/Tian2020MSA_2023/'):

        # Init.
        self.cortex = cortex
        self.cortex_res = cortex_res
        self.scale = scale
        network_details = None

        if cortex_res == None:
            self.label = 'TianS'+str(scale)+cortex
        else:
            self.label = 'TianS'+str(scale)+cortex+str(cortex_res)

        if cortex == 'Glasser':
            file = path+'3T/Cortex-Subcortex/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR_Tian_Subcortex_S' + \
                str(scale)+'.dlabel.nii'
            details = path+'3T/Cortex-Subcortex/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR_Tian_Subcortex_S' + \
                str(scale)+'_label.txt'
            network_details = '/home/lukeh/LabData/Lab_LucaC/Luke/Backups/hpc_backups/parcellations/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey.txt'
            file_volume = None

        elif cortex == 'Schaefer':
            file = (path+'3T/Cortex-Subcortex/'
                    + 'Schaefer2018_'+str(cortex_res)+'Parcels_17Networks_order_Tian_Subcortex_S'+str(scale)+'.dlabel.nii')
            details = (path+'3T/Cortex-Subcortex/'
                       + 'Schaefer2018_'+str(cortex_res)+'Parcels_17Networks_order_Tian_Subcortex_S'+str(scale)+'_label.txt')
            file_volume = (path+'3T/Cortex-Subcortex/MNIvolumetric/'
                           + 'Schaefer2018_'+str(cortex_res)+'Parcels_17Networks_order_Tian_Subcortex_S'+str(scale)+'_3T_MNI152NLin2009cAsym_1mm.nii.gz')

        elif cortex == 'Schaefer200Networks':
            file = '../data/Schaefer2018_200_Networks.dlabel.nii'
            details = None
            file_volume = None

        else:
            print('Parcellation not found...')

        # the parcellation file
        if volumetric:
            self.file = file_volume
        else:
            self.file = file

        # text file associated with parcellation
        self.details = details
        self.network_details = network_details

        # the actual image
        self.img = nb.load(self.file).get_fdata().reshape(-1, 1)

        # number of parcels
        self.n_parcels = len(np.unique(self.img)) - 1  # account for '0'

        # anatomical labels of parcels
        if details is not None:
            try:
                parcel_labels = np.loadtxt(self.details, dtype=str)

            except:
                parcel_labels = []
                with open(self.details, 'r') as f:
                    for count, line in enumerate(f, start=0):
                        if count % 2 == 0:
                            parcel_labels.append(line.split('\n')[0])
                parcel_labels = np.array(parcel_labels)
            self.parcel_labels = parcel_labels
        else:
            self.parcel_labels = list(range(self.n_parcels))
        assert len(
            self.parcel_labels) == self.n_parcels, "Mismatch between labels and parc"

    def apply_networks(self, parcel_list_input):
        if self.cortex == "Glasser":
            network_df = pd.read_csv(self.network_details,
                                     delimiter='\t')
            network_labels = []
            # Match the network labels to roi indices
            for parcel in parcel_list_input:
                label = network_df[network_df.GLASSERLABELNAME ==
                                   parcel].NETWORK.values
                if label.size == 1:
                    label = label[0]
                elif label.size == 0:
                    label = 'n/a'
                network_labels.append(label)

        elif self.cortex == "Schaefer":
            network_labels = []
            for i in parcel_list_input:
                try:
                    network_labels.append(i.split("_")[2])
                except:
                    network_labels.append("n/a")

        else:
            print("function not available")

        return network_labels
