"""Publication-quality brain surface plotting utilities.

Provides 4-panel brain visualisations (lateral/medial × left/right) with
optional contour overlays for displaying RSA and decoding results on the
cortical surface. Used by the analysis notebooks for figure generation.
"""
import nibabel as nb
import seaborn as sns
import hcp_utils as hcp
import nilearn.plotting as plotting
import matplotlib.pyplot as plt
import numpy as np
import cmasher as cmr


def pub_surf_plot(input_array, parcellation, save=None, size=(2.2, 2.2), thresh=None, vmin=None, vmax=None, cmap="viridis"):
    if vmax is None:
        vmax = np.nanmax(input_array)
    if vmin is None:
        vmin = np.nanmin(input_array)
    parc = np.squeeze(nb.load(parcellation).get_fdata())
    output = np.zeros((parc.shape))
    borders = np.zeros((parc.shape))
    parcels = np.unique(parc)
    parcels = np.delete(parcels, 0)  # assume 0 denotes no roi

    for i in parcels:
        index = (parc == i)
        output[index] = input_array[int(i)-1]
        borders[index] = i

    fig, axs = plt.subplot_mosaic("""
                              AB
                              CD
                              """,
                                  figsize=size, subplot_kw=dict(
                                      projection='3d'),
                                  constrained_layout=True)
    count = 0
    ax_list = ["A", "B", "C", "D"]
    darkness = 1.0
    alpha = 1.0
    for view in ['lateral', 'medial']:
        for hemi in ['left', 'right']:

            if hemi == 'left':
                surface = hcp.left_cortex_data(output)

                mesh = hcp.mesh.inflated_left
                bg_map = hcp.mesh.sulc_left

            elif hemi == 'right':
                surface = hcp.right_cortex_data(output)

                mesh = hcp.mesh.inflated_right
                bg_map = hcp.mesh.sulc_right

            if save is not None:
                plt.close("all")
                plotting.plot_surf_stat_map(mesh, surface, bg_map=bg_map, view=view, cmap=cmap, threshold=abs(thresh) if thresh is not None else None,
                                            hemi=hemi, bg_on_data=True, darkness=darkness,
                                            alpha=alpha, vmax=vmax, vmin=vmin, colorbar=False, symmetric_cbar=False)
                plt.savefig(save+'_'+hemi+'_'+view+'.jpeg', dpi=600)
                plt.close("all")
                plotting.plot_surf_stat_map(mesh, surface, bg_map=bg_map, view=view, cmap=cmap, threshold=abs(thresh) if thresh is not None else None,
                                            hemi=hemi, bg_on_data=True, darkness=darkness,
                                            alpha=alpha, vmax=vmax, vmin=vmin, colorbar=True, symmetric_cbar=False)
                plt.savefig(save+'_'+hemi+'_'+view+'_colorbar.jpeg', dpi=600)
                plt.close("all")

            else:

                ax = axs[ax_list[count]]

                plotting.plot_surf_stat_map(mesh, surface, bg_map=bg_map, view=view, cmap=cmap, threshold=abs(thresh) if thresh is not None else None,
                                            hemi=hemi, bg_on_data=True, darkness=darkness, axes=ax,
                                            alpha=alpha, vmax=vmax, vmin=vmin, colorbar=True, symmetric_cbar=False)

            count = count+1

    plt.show()


def pub_surf_plot_contours(input_array, input_array2, parcellation, save=None, size=(2.2, 2.2), thresh=None, vmin=None, vmax=None, cmap="viridis"):
    if vmax is None:
        vmax = np.nanmax(input_array)
    if vmin is None:
        vmin = np.nanmin(input_array)
    parc = np.squeeze(nb.load(parcellation).get_fdata())
    output = np.zeros((parc.shape))
    output2 = np.zeros((parc.shape))
    #borders = np.zeros((parc.shape))
    parcels = np.unique(parc)
    parcels = np.delete(parcels, 0)  # assume 0 denotes no roi

    for i in parcels:
        index = (parc == i)
        output[index] = input_array[int(i)-1]
        output2[index] = input_array2[int(i)-1]
        #borders[index] = i

    fig, axs = plt.subplot_mosaic("""
                              AB
                              CD
                              """,
                                  figsize=size, subplot_kw=dict(
                                      projection='3d'),
                                  constrained_layout=True)
    count = 0
    ax_list = ["A", "B", "C", "D"]
    darkness = 1.0
    alpha = 1.0
    for view in ['lateral', 'medial']:
        for hemi in ['left', 'right']:

            if hemi == 'left':
                surface = hcp.left_cortex_data(output)
                surface2 = hcp.left_cortex_data(output2)
                mesh = hcp.mesh.inflated_left
                bg_map = hcp.mesh.sulc_left

            elif hemi == 'right':
                surface = hcp.right_cortex_data(output)
                surface2 = hcp.right_cortex_data(output2)
                mesh = hcp.mesh.inflated_right
                bg_map = hcp.mesh.sulc_right

            if save is not None:
                plt.close("all")
                plotting.plot_surf_stat_map(mesh, surface, bg_map=bg_map, view=view, cmap=cmap, threshold=abs(thresh) if thresh is not None else None,
                                            hemi=hemi, bg_on_data=True, darkness=darkness,
                                            alpha=alpha, vmax=vmax, vmin=vmin, colorbar=False, symmetric_cbar=False)
                ax = plt.gca()
                for roi in np.unique(surface2)[1::]:
                    try:
                        plotting.plot_surf_contours(mesh, surface2, levels=[int(roi)], bg_map=bg_map, view=view, colors='k', hemi=hemi, bg_on_data=True, darkness=0.50, axes=ax)
                    except:
                        print("contour not plotting", hemi, view)
                plt.savefig(save+'_'+hemi+'_'+view+'.jpeg', dpi=600)
                plt.close("all")
                plotting.plot_surf_stat_map(mesh, surface, bg_map=bg_map, view=view, cmap=cmap, threshold=abs(thresh) if thresh is not None else None,
                                            hemi=hemi, bg_on_data=True, darkness=darkness,
                                            alpha=alpha, vmax=vmax, vmin=vmin, colorbar=True, symmetric_cbar=False)
                plt.savefig(save+'_'+hemi+'_'+view+'_colorbar.jpeg', dpi=600)
                plt.close("all")

            else:
                ax = axs[ax_list[count]]

                plotting.plot_surf_stat_map(mesh, surface, bg_map=bg_map, view=view, cmap=cmap, threshold=abs(thresh) if thresh is not None else None,
                                            hemi=hemi, bg_on_data=True, darkness=darkness, axes=ax,
                                            alpha=alpha, vmax=vmax, vmin=vmin, colorbar=False, symmetric_cbar=False)
                
                for roi in np.unique(surface2)[1::]:
                    try:
                        plotting.plot_surf_contours(mesh, surface2, levels=[int(roi)], bg_map=bg_map, view=view, colors='k', hemi=hemi, bg_on_data=True, darkness=0.50, axes=ax)
                    except:
                        print("contour not plotting", hemi, view)

            count = count+1

    plt.show()

def get_cropped_img(img,
                    x_start_buffer=10,
                    x_finish_buffer=10,
                    y_start_buffer=10,
                    y_finish_buffer=10,
                    ):
    # Returns a cropped input image

    # average across color information (3rd dimension)
    m_img = np.mean(img, axis=2)

    x_start = (np.where(np.argmax(m_img != np.max(img), axis=0))
               [0][0]) - x_start_buffer
    x_finish = (np.where(np.argmax(m_img != np.max(img), axis=0))
                [0][-1]) + x_finish_buffer
    y_start = np.where(np.argmax(m_img != np.max(img), axis=1))[
        0][0] - y_start_buffer
    y_finish = np.where(np.argmax(m_img != np.max(img), axis=1))[
        0][-1] + y_finish_buffer

    # Crop the input image based on determined crop heights and widths
    img_cropped = img[y_start:y_finish, x_start:x_finish, :]
    return img_cropped
