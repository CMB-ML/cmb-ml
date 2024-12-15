from pathlib import Path
import requests
import logging

import numpy as np
from tqdm import tqdm
from get_data.utils.new_download_utils import (
    download,
    download_progress,
)


logger = logging.getLogger(__name__)


def format_freq(freq):
    return "{:.0f}".format(freq).zfill(3)
def format_real(real):
    return "{:.0f}".format(real).zfill(5)


def get_planck_obs_data_ext(detector, assets_directory, progress=False):
    planck_obs_fn = "{instrument}_SkyMap_{frequency}_{obs_nside}_{rev}_full.fits"
    url_template_maps = "http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID={fn}"
    # Setup to get maps... this is all naming convention stuff
    if detector in [30, 44, 70]:
        instrument = "LFI"
        use_freq_str = format_freq(detector) + "-BPassCorrected"
        rev = "R3.00"
        obs_nside = 1024
    else:
        instrument = "HFI"
        use_freq_str = format_freq(detector)
        rev = "R3.01"
        obs_nside = 2048
    if detector == 353:
        use_freq_str = format_freq(detector) + "-psb"

    obs_map_fn = planck_obs_fn.format(instrument=instrument, frequency=use_freq_str, rev=rev, obs_nside=obs_nside)
    dest_path = Path(assets_directory) / obs_map_fn
    if progress:
        if detector in [30, 44, 70]:
            file_size = 502       # IQU maps at nside=1024 are 503 MB
        elif detector in [545, 857]:
            file_size = 603       # I maps at nside=2048 are 604 MB
        else:                     # 100, 143, 217, 353
            file_size = 2 * 1000  # IQU maps at nside=2048 are 2 GB

        download_progress(dest_path, url_template_maps, file_size=file_size)
    else:
        download(dest_path, url_template_maps)

    return dest_path


def get_planck_hm_data_ext(detector, assets_directory, progress=False):
    hm_map_fn_template = "{instrument}_SkyMap_{freq}_2048_R3.01_halfmission-{hm}.fits"
    url_template_maps = "http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID={fn}"

    if detector in [30, 44, 70]:
        instrument = "LFI"
    else:
        instrument = "HFI"
    hm_1_fn = hm_map_fn_template.format(instrument=instrument, freq=format_freq(detector), hm=1)
    hm_2_fn = hm_map_fn_template.format(instrument=instrument, freq=format_freq(detector), hm=2)

    hm_1_fn = Path(assets_directory) / hm_1_fn
    hm_2_fn = Path(assets_directory) / hm_2_fn

    if progress:
        if detector in [30, 44, 70]:
            file_size = 2 * 1000  # IQU maps at nside=1024
        elif detector in [545, 857]:
            file_size = 603       # I maps at nside=2048
        else:                     # 100, 143, 217, 353
            file_size = 2 * 1000  # IQU maps at nside=2048
        download_progress(hm_1_fn, url_template_maps, file_size=file_size)
        download_progress(hm_2_fn, url_template_maps, file_size=file_size)
    else:
        download(hm_1_fn, url_template_maps)
        download(hm_2_fn, url_template_maps)
    return hm_1_fn, hm_2_fn


def get_planck_noise_fn(detector, realization):
    ring_cut = "full"
    planck_noise_fn_template = "ffp10_noise_{frequency}_{ring_cut}_map_mc_{realization}.fits"

    fn = planck_noise_fn_template.format(frequency=format_freq(detector), 
                                         ring_cut=ring_cut, 
                                         realization=format_real(realization))
    return fn


def get_planck_noise_data_ext(detector, assets_directory, realization=0, progress=False):
    """
    Get the filename for the Planck noise data, downloading it if necessary.

    Parameters
    ----------
    realization : int
        The realization number for the noise map. Default is 0. There are 300 available.
    """
    # All file sizes are in decimal MB, as seen in Files explorer, minus 1
    if detector in [30, 44, 70]:
        file_size = 150  # IQU maps at nside=1024
    elif detector in [545, 857]:
        file_size = 200  # I maps at nside=2048
    else:                # 100, 143, 217, 353
        file_size = 603  # IQU maps at nside=2048

    fn = Path(assets_directory) / get_planck_noise_fn(detector, realization)
    url_template_sims = "http://pla.esac.esa.int/pla/aio/product-action?SIMULATED_MAP.FILE_ID={fn}"
    if progress:
        download_progress(fn, url_template_sims, file_size=file_size)
    else:
        download(fn, url_template_sims)
    return fn


def get_planck_pred_data_ext(detector, assets_directory, realization=0):
    """
    Get the filename for the Planck noise data, downloading it if necessary.

    Parameters
    ----------
    realization : int
        The realization number for the noise map. Default is 0. There are 300 available.
    """
    planck_noise_fn_template = "COM_CMB_IQU-nilc_2048_R3.00_full.fits"
    url_template_sims = "http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID={fn}"

    fn = planck_noise_fn_template.format(frequency=format_freq(detector), 
                                         realization=format_real(realization))
    fn = Path(assets_directory) / fn
    download(fn, url_template_sims)
    return fn
