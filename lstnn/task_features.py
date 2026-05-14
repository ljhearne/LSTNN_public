import numpy as np
from scipy.spatial.distance import jaccard
import rsatoolbox


def calc_jaccard(data):
    mat = np.zeros((108, 108))
    for i in range(108):
        for j in range(108):
            mat[i, j] = jaccard(data[i, :], data[j, :])
    return mat


def get_experimentor_rdms(LST_puzzle_ds):
    # visual input
    data = np.zeros((108, 80))
    for i in range(108):
        data[i, :] = LST_puzzle_ds[i][0].reshape(-1).numpy()

    # create description dict
    puzzles = ['LST' + str(i) for i in range(1, 108+1)]
    obs_des = {'puzzles': puzzles}

    # visual input
    mat = calc_jaccard(data)
    visual_rdm = rsatoolbox.rdm.RDMs(
        dissimilarities=mat.reshape(1, 108, 108),
        dissimilarity_measure='jaccard',
        rdm_descriptors={'name': "visual"},
        pattern_descriptors=obs_des,
    )

    # motor output, i.e., correct button presses
    data = np.zeros((108, 4))
    for i in range(108):
        data[i, :] = LST_puzzle_ds[i][1].reshape(-1).numpy()

    mat = calc_jaccard(data)
    motor_rdm = rsatoolbox.rdm.RDMs(
        dissimilarities=mat.reshape(1, 108, 108),
        dissimilarity_measure='jaccard',
        rdm_descriptors={'name': "motor"},
        pattern_descriptors=obs_des,
    )

    # Relational complexity
    data = np.ones((108, 108))
    data[0:36, 0:36] = 0
    data[36:72, 36:72] = 0
    data[72::, 72::] = 0
    data[0:36, 36:72] = 0.5
    data[36:72, 72::] = 0.5
    condition_rdm = rsatoolbox.rdm.RDMs(data.reshape(1, 108, 108),
                                        rdm_descriptors={'name': "complexity"},
                                        dissimilarity_measure='experimentor')

    output = []
    output.append(visual_rdm)
    output.append(condition_rdm)
    output.append(motor_rdm)
    return output