import healpy as hp
import matplotlib.pyplot as plt
from cmbml.utils.planck_cmap import colombi1_cmap
import numpy as np
from itertools import product



def show_map(combo):
    masking, deconv, remove_dipole, working_dir, mask_output = combo
    map_fn1 = "/data/jim/CMB_Data/Datasets/I_512_1450/{}/Analysis_D_Common_Post_Map_Real/Test/sim0000/cmb_real_post.fits"
    map_fn2 = "/data/jim/CMB_Data/Datasets/I_512_1450/{}/Analysis_E_Common_Post_Map_Pred/Test/sim0000/cmb_pred_post.fits"
    mask_fn = "/data/jim/CMB_Data/Datasets/I_512_1450/Simulation_Mask/mask.fits"

    dconv_str = "ydec" if deconv else "ndec"
    remove_dipole_str = "yrd" if remove_dipole else "nrd"

    map_fn1 = map_fn1.format(working_dir, masking, dconv_str, remove_dipole_str)
    map_fn2 = map_fn2.format(working_dir, masking, dconv_str, remove_dipole_str)

    data1 = hp.read_map(map_fn1)
    data2 = hp.read_map(map_fn2)
    mask = hp.read_map(mask_fn)

    diff = data1 - data2
    diff_masked = hp.ma(diff)
    diff_masked.mask = np.logical_not(mask)

    if mask_output:
        diff = diff_masked

    mask_strs = {"hpma": "Masked Array", "yxma": "Set to Zero", "noma": "Unmasked"}
    conv_strs = {True: "Deconvolved", False: "Not Deconvolved"}
    dipole_strs = {True: "Dipole Removed", False: "Dipole Not Removed"}
    title = f"{working_dir}, {mask_strs[masking]}, {conv_strs[deconv]}, {dipole_strs[remove_dipole]}"

    mask_out_str = "yfm" if mask_output else "nfm"

    fn = f"map_figs/{mask_out_str}_{working_dir}_{masking}_{dconv_str}_{remove_dipole_str}.png"

    fig, [ax1, ax2, ax3, ax4] = plt.subplots(1, 4, figsize=(20, 5))

    plt_params = dict(cmap=colombi1_cmap, hold=True)
    plt_params_abs = {**plt_params,  "min": -400, "max": 400}
    plt_params_diff = {**plt_params, "min": -120, "max": 120}

    plt.axes(ax1)
    hp.mollview(data1, title="Real", **plt_params_abs)
    plt.axes(ax2)
    hp.mollview(data2, title="Pred", **plt_params_abs)
    plt.axes(ax3)
    hp.mollview(diff, title="Difference", **plt_params_diff)
    plt.axes(ax4)
    plt.hist(diff_masked.compressed(), bins=100, range=(-120, 120))
    ax4.set_yticks([])
    plt.suptitle(title)
    plt.savefig(fn)
    plt.close()


maskings = ["hpma"]
deconvs = [True]
remove_dipoles = [True]
# maskings = ["hpma", "yxma", "noma"]
# deconvs = [True, False]
# remove_dipoles = [True, False]
working_dirs = ["CMBNNCS", "PyILC_CNILC"]
mask_outputs = [True, False]

combos = list(product(maskings, deconvs, remove_dipoles, working_dirs, mask_outputs))
for combo in combos:
    print(f"Doing {combo}")
    show_map(combo)
