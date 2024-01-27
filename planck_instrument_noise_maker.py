from pathlib import Path
import numpy as np

import healpy as hp

from astropy.io import fits
import pysm3.units as u
from astropy.cosmology import Planck15


FITS_HEADER = 1


class NoiseCacheCreator:
    def __init__(self, source_path: Path, nside: int, center_frequency):
        self.hdr: int = FITS_HEADER
        self.ref_path = source_path
        self.nside = nside
        self.center_frequency = center_frequency

    def create_noise_cache(self, field_str, cache_path) -> None:
        field_idx = self._get_field_idx(field_str)  # To be set when fits file is open
        m = self._get_map_at_nside(field_idx)
        
        # Convert variance to standard deviation, 
        #   to be used in np.random.Generator.normal as "scale" parameter
        m = np.sqrt(m)

        # Convert MJy/sr to K_CMB (I think, TODO: Verify)
        # This is an oversimplification applied to the 545 and 857 GHz bands
        # something about "very sensitive to band shape" for sub-mm bands (forgotten source)
        # This may be a suitable first-order approximation
        if self._get_map_unit_sqrt(field_idx) == "MJy/sr":
            m = (m * u.MJy / u.sr).to(
                u.K, equivalencies=u.thermodynamic_temperature(self.center_frequency, Planck15.Tcmb0)
            ).value

        col_names = {"T": "II", "Q": "QQ", "U": "UU"}
        hp.write_map(filename=str(cache_path),
                     m=m,
                     nest=False,
                     column_names=[col_names[field_str]],
                     column_units=["K_CMB"],
                     dtype=m.dtype,
                     overwrite=True
                    # TODO: figure out how to add a comment to hp's map... or switch with astropy equivalent 
                    #  extra_header=f"Variance map pulled from {self.ref_map_fn}, {col_names[field_str]}"
                     )
        return m

    def _get_field_idx(self, field_str):
        n_fields_in_ref_file = self._get_number_fields_in_ref_file()
        field_idx = self._lookup_ref_map_var_field(field_str, n_fields_in_ref_file)
        return field_idx

    def _get_number_fields_in_ref_file(self):
        with fits.open(self.ref_path) as hdul:
            n_fields = len(hdul[self.hdr].columns)
        return n_fields

    def _lookup_ref_map_var_field(self, field_str, n_fields):
        T_n_fields = 3
        TQU_n_fields = 10
        
        if n_fields == T_n_fields:
            if field_str == "T":
                field_idx = 2
            else:
                raise ValueError("Detector only has 'T' field.")
        elif n_fields == TQU_n_fields:
            if field_str == "T":
                field_idx = 4
            elif field_str == "Q":
                field_idx = 7
            elif field_str == "U":
                field_idx = 9
            else:
                raise ValueError(f"Field is {field_str}, expected 'T', 'Q', or 'U'")
        else:
            raise ValueError(f"Unexpected number of fields in fits file.")
        return field_idx    

    def _get_map_at_nside(self, field_idx):
        m = hp.read_map(self.ref_path, field=field_idx)
        # From PySM3 template.py's read_map function, with minimal alteration:
        dtype = m.dtype
        # numba only supports little endian
        if dtype.byteorder == ">":
            dtype = dtype.newbyteorder()
        # mpi4py has issues if the dtype is a string like ">f4"
        if dtype == np.dtype(np.float32):
            dtype = np.dtype(np.float32)
        elif dtype == np.dtype(np.float64):
            dtype = np.dtype(np.float64)
        
        nside_in = hp.get_nside(m)
        if self.nside < nside_in:  # do downgrading in double precision
            m = hp.ud_grade(m.astype(np.float64), nside_out=self.nside)
        elif self.nside > nside_in:
            m = hp.ud_grade(m, nside_out=self.nside)
        m = m.astype(dtype, copy=False)
        # End of code from PySM3 template.py
        return m

    def _get_map_unit_sqrt(self, field_idx):
        # Can't use PySM3's read_map() function
        # I don't know how to make the units "(K_CMB)^2" parseable by astropy's units module
        with fits.open(self.ref_path) as hdul:
            try:
                field_num = field_idx + 1
                unit = hdul[self.hdr].header[f"TUNIT{field_num}"]
            except KeyError:
                unit = ""
                # TODO: Use logging
                # log.warning("No physical unit associated with file %s", str(path))
        ok_units_k_cmb = ["(K_CMB)^2", "Kcmb^2"]
        ok_units_mjysr = ["(Mjy/sr)^2"]
        ok_units = [*ok_units_k_cmb, *ok_units_mjysr]
        # TODO: Use logging
        # log.info(f"Units for map {self.ref_map_fn} are {unit}")
        if unit not in ok_units:
            raise ValueError(f"Wrong unit found in fits file. Found {unit}, expected one of {ok_units}.")
        if unit in ok_units_k_cmb:
            unit_sqrt = "K_CMB"
        else:
            unit_sqrt = "MJy/sr"
        return unit_sqrt


def random_noise_map_maker(sd_map, random_seed, center_frequency):
    rng = np.random.default_rng(random_seed)
    noise_map = rng.normal(scale=sd_map, size=sd_map.size)
    noise_map = u.Quantity(noise_map, u.K_CMB, copy=False)
    noise_map = noise_map.to(u.uK_RJ, equivalencies=u.cmb_equivalencies(center_frequency))
    return noise_map
