from pathlib import Path
from pprint import pprint
import numpy as np
import healpy as hp
import pysm3
import pysm3.units as u
import matplotlib.pyplot as plt
from planck_cmap import colombi1_cmap
from cmb_component_old import create_cmb_lensed_from_params

from cosmo_params import get_params_from_wmap_chains


def try_create_cmb_lensed_from_params(params):
    nside = 512
    lmax = 8150
    cmb_seed = 0
    # cmb_ps_file_to_save = "test_ps.txt"
    cmb_ps_file_to_save = None

    # From planck_deltabandpass.tbl
    detector_frq = 100.89 * u.GHz
    fwhm = 9.682 * u.arcmin

    cmb = create_cmb_lensed_from_params(cosmo_params=params,
                                        cmb_ps_file_out=cmb_ps_file_to_save,
                                        lmax=lmax,
                                        nside=nside,
                                        cmb_seed=cmb_seed,
                                        apply_delens=False
                                        )
    sky = pysm3.Sky(nside=nside, component_objects=[cmb], output_unit="uK_RJ")
    skymaps = sky.get_emission(detector_frq)
    skymap = skymaps[0]
    map_smoothed = pysm3.apply_smoothing_and_coord_transform(skymap, fwhm=fwhm)

    hp.mollview(map_smoothed, min=-300, max=300,
                title=f"Test Map {detector_frq}", 
                unit=map_smoothed.unit,
                cmap=colombi1_cmap)
    plt.show()


def try_create_cmb_lensed_from_basic_params():
    # Params from camb demo file https://camb.readthedocs.io/en/latest/CAMBdemo.html
    # params = dict(H0=67.4, 
    #               ombh2=0.0224, 
    #               omch2=0.120, 
    #               mnu=0.06, 
    #               omk=0.001, 
    #               tau=0.054,
    #               As=2e-9,
    #               ns=0.965
    #               )

    # Params from WMAP chains, hardcoded for test
    params = dict(H0=70.63212047223567, 
                  ombh2=0.022918285755026139, 
                  omch2=0.113033177236981461, 
                  tau=0.098830384765900753,
                  ns=0.982695162357922314
                  )
    pprint(params)
    try_create_cmb_lensed_from_params(params)


def try_create_cmb_lensed_from_wmap_params():
    wmap_path = Path("wmap_assets/wmap_lcdm_wmap9_chains_v5")
    params_to_get = ['H0', 'omegach2', 'omegabh2', 'ns002', 'tau']
    n_vals = 2
    rng = np.random.default_rng(seed=0)
    res = get_params_from_wmap_chains(wmap_path, n_vals, params_to_get, rng)
    single_param_set = res.get_idx(1)
    pprint(single_param_set.camb_format())
    try_create_cmb_lensed_from_params(single_param_set.camb_format())


if __name__ == "__main__":
    # try_create_cmb_lensed_from_basic_params()
    try_create_cmb_lensed_from_wmap_params()
