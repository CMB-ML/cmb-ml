from pathlib import Path
import logging

import utils.fits_inspection as fits_inspect
from ..physics_instrument_noise import planck_result_to_sd_map

import healpy as hp


logger = logging.getLogger(__name__)

class NoiseCacheCreator:
    def __init__(self, source_path: Path, nside: int, center_frequency):
        self.hdu: int = fits_inspect.ASSUME_FITS_HEADER
        self.ref_path = source_path
        self.nside = nside
        self.center_frequency = center_frequency

    def create_noise_cache(self, field_str, cache_path) -> None:
        logger.debug(f"component_instrument_noise.create_noise_cache start")
        field_idx = fits_inspect.lookup_field_idx(field_str, self.ref_path, self.hdu)
        
        st_dev_skymap = planck_result_to_sd_map(fits_fn=self.ref_path, 
                                                hdu=self.hdu,
                                                field_idx=field_idx, 
                                                nside_out=self.nside,
                                                cen_freq=self.center_frequency)

        col_names = {"T": "II", "Q": "QQ", "U": "UU"}
        hp.write_map(filename=str(cache_path),
                     m=st_dev_skymap,
                     nest=False,
                     column_names=[col_names[field_str]],
                     column_units=["K_CMB"],
                     dtype=st_dev_skymap.dtype,
                     overwrite=True
                    # TODO: figure out how to add a comment to hp's map... or switch with astropy equivalent 
                    #  extra_header=f"Variance map pulled from {self.ref_map_fn}, {col_names[field_str]}"
                     )
        logger.debug(f"component_instrument_noise.create_noise_cache end")
        return st_dev_skymap


class Detector:
    def __init__(self, nom_freq: int, cen_freq, fwhm):
        self.nom_freq = nom_freq
        self.cen_freq = cen_freq
        self.fwhm = fwhm


