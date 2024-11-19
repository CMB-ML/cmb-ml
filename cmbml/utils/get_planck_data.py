from pathlib import Path
import requests
import logging

import numpy as np
from tqdm import tqdm


logger = logging.getLogger(__name__)

def acquire_map_data_progress(dest_path, source_url_template, file_size=None):
    """Load map data from a file, downloading it if necessary."""
    need_to_dl = check_need_download(dest_path, file_size=file_size)
    fn = dest_path.name
    if need_to_dl:
        response = requests.get(source_url_template.format(fn=fn), stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        if total_size == 0 and file_size is not None:
            total_size = file_size * 1000 * 1000  # Convert MB to bytes

        chunk_size = 1024 * 1024  # Download in 1MB chunks

        with open(dest_path, "wb") as file, tqdm(
            desc=f"Downloading {fn}",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            leave=False,
            position=1
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                file.write(chunk)
                progress_bar.update(len(chunk))


def acquire_map_data(dest_path, source_url_template):
    """Load map data from a file, downloading it if necessary."""
    need_to_dl = check_need_download(dest_path)
    fn = dest_path.name
    if need_to_dl:
        response = requests.get(source_url_template.format(fn=fn))
        with open(dest_path, "wb") as file:
            file.write(response.content)
        print(f"Downloaded {fn}")


def check_need_download(dest_path, file_size=None):
    """Check if a file exists."""
    need_to_dl = False
    if not dest_path.exists():
        logger.info(f"File {dest_path} does not exist; downloading.")
        need_to_dl = True
    elif dest_path.stat().st_size < 1024:  # If the file is less than 1KB, it's a placeholder file
        logger.info(f"File {dest_path} has placeholder file; redownloading.")
        need_to_dl = True
    elif file_size is not None and dest_path.stat().st_size < file_size * 1000 * 1000:
        logger.info(f"File {dest_path} is too small; redownloading.")
        need_to_dl = True
    else:
        logger.debug(f"File {dest_path} exists.")
    return need_to_dl


def format_freq(freq):
    return "{:.0f}".format(freq).zfill(3)
def format_real(real):
    return "{:.0f}".format(real).zfill(5)


def get_planck_obs_data(detector, assets_directory):
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
    acquire_map_data(dest_path, url_template_maps)  # Download the data if it doesn't exist. Do this ahead of time.
    return dest_path


def get_planck_hm_data(detector, assets_directory, progress=False):
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
        acquire_map_data_progress(hm_1_fn, url_template_maps, file_size=file_size)
        acquire_map_data_progress(hm_2_fn, url_template_maps, file_size=file_size)
    else:
        acquire_map_data(hm_1_fn, url_template_maps)
        acquire_map_data(hm_2_fn, url_template_maps)
    return hm_1_fn, hm_2_fn


def get_planck_noise_fn(detector, realization):
    ring_cut = "full"
    planck_noise_fn_template = "ffp10_noise_{frequency}_{ring_cut}_map_mc_{realization}.fits"

    fn = planck_noise_fn_template.format(frequency=format_freq(detector), 
                                         ring_cut=ring_cut, 
                                         realization=format_real(realization))
    return fn


def get_planck_noise_data(detector, assets_directory, realization=0, progress=False):
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
        acquire_map_data_progress(fn, url_template_sims, file_size=file_size)
    else:
        acquire_map_data(fn, url_template_sims)
    return fn


# def get_map_dtype(m: np.ndarray):
#     """
#     Get the data type of a map in a format compatible
#     with numba and mpi4py.

#     Args:
#         m (np.ndarray): Numpy array representing the map.

#     Returns:
#         np.dtype: The data type of the map.
#     """
#     # From PySM3 template.py's read_map function, with minimal alteration:
#     dtype = m.dtype
#     # numba only supports little endian
#     if dtype.byteorder == ">":
#         dtype = dtype.newbyteorder()
#     # mpi4py has issues if the dtype is a string like ">f4"
#     if dtype == np.dtype(np.float32):
#         dtype = np.dtype(np.float32)
#     elif dtype == np.dtype(np.float64):
#         dtype = np.dtype(np.float64)
#     # End of used portion
#     return dtype
