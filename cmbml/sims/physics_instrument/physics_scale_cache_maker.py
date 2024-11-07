import logging

import numpy as np
from astropy.units import Quantity, Unit
import healpy as hp
import pysm3.units as u

import cmbml.utils.fits_inspection as fits_inspect
from cmbml.utils.fits_inspection import get_num_all_fields_in_hdr
from cmbml.utils.physics_units import convert_field_str_to_Unit


logger = logging.getLogger(__name__)


class ScaleCacheMaker:
    """
    Class to create a cache for the noise maps. Scale is the standard deviation of the variance maps.
    """
    def __init__(self, cfg, name_tracker, in_varmap_source, out_scale_cache):
        self.cfg = cfg
        self.name_tracker = name_tracker
        self.in_noise_src = in_varmap_source
        self.nside_out = cfg.scenario.nside
        self.out_scale_cache = out_scale_cache

    def get_src_path(self, detector: int):
        """
        Get the path for the source noise file based on the hydra configs.

        Parameters:
        detector (int): The nominal frequency of the detector.

        Returns:
        str: The path for the fits file containing the noise.
        """
        fn       = self.cfg.model.sim.noise.src_files[detector]
        src_root = self.cfg.local_system.assets_dir
        contexts_dict = dict(src_root=src_root, filename=fn)
        with self.name_tracker.set_contexts(contexts_dict):
            src_path = self.in_noise_src.path
        return src_path

    def get_field_idx(self, src_path, field_str) -> int:
        """
        Looks at fits file to determine field_idx corresponding to field_str

        Parameters:
        src_path (str): The path to the fits file.
        field_str (str): The field string to look up.

        Returns:
        int: The field index corresponding to the field string.
        """
        hdu = self.cfg.model.sim.noise.hdu_n
        field_idcs_dict = dict(self.cfg.model.sim.noise.field_idcs)
        # Get number of fields in map
        n_map_fields = get_num_all_fields_in_hdr(fits_fn=src_path, hdu=hdu)
        # Lookup field index based on config file
        field_idx = field_idcs_dict[n_map_fields][field_str]
        return field_idx
    
    def make_cache_for_freq(self, freq, detector, hdu):
        """
        Creates a cache for the given frequency and detector.

        Parameters:
        freq (int): The frequency of the detector.
        detector (int): The detector number.

        Returns:
        None
        """
        src_path = self.get_src_path(freq)

        field_idcs = [self.get_field_idx(src_path, field_str) for field_str in detector.fields]
        st_dev_skymap = planck_result_to_sd_map(nside_out=self.nside_out,
                                                fits_fn=src_path,
                                                hdu=hdu,
                                                field_idx=field_idcs,
                                                cen_freq=detector.cen_freq)

        with self.name_tracker.set_contexts(dict(freq=freq)):
            self.write_wrapper(data=st_dev_skymap, fields=detector.fields)

    def write_wrapper(self, data: Quantity, fields: str):
        """
        Wraps the write method for the noise cache asset to ensure proper column names and units.

        Parameters:
        data (Quantity): The standard deviation map to write to the noise cache.
        field_str (str): The field string (either T or TQU).
        """
        col_names = [field_ch + "_STOKES" for field_ch in fields]
        units = [data.unit for _ in fields]

        # We want to give some indication that for I field, this is from the II covariance (or QQ, UU)
        logger.debug(f'Writing NoiseCache map to path: {self.out_scale_cache.path}')
        self.out_scale_cache.write(data=data.value,
                                   column_names=col_names,
                                   column_units=units,
                                   extra_header=[("METHOD", "FROM_VAR", "Sqrt Planck obs, fields II_, QQ_, UU_COV")])
        # TODO: Test load this file; see if column names and units match expectation.
        logger.debug(f'Wrote NoiseCache map to path: {self.out_scale_cache.path}')


def make_random_noise_map(sd_map, random_seed):
    """
    Make a random noise map. 

    Args:
        sd_map (np.ndarray): The standard deviation map created with planck_result_to_sd_map.
        random_seed (int): The seed for the random number generator.
        center_frequency (float): The center frequency of the detector.
    """
    #TODO: set units when redoing this function
    rng = np.random.default_rng(random_seed)
    noise_map = rng.normal(scale=sd_map)
    noise_map = u.Quantity(noise_map, u.K_CMB, copy=False)
    return noise_map


def planck_result_to_sd_map(nside_out, fits_fn, hdu, field_idx, cen_freq):
    """
    Convert a Planck variance map to a standard deviation map, with units of K_CMB.

    In the observation maps provided by Planck, fields 0,1,2 are stokes 
    parameters for T, Q, and U (resp). The HITMAP is field 3. The remainder
    are variance maps: II, IQ, IU, QQ, QU, UU. We use variance maps to generate
    noise maps, albeit with the simplification of ignoring covariance 
    between the stokes parameters.

    Args:
        fits_fn (str): The filename of the fits file.
        hdu (int): The HDU to read.
        field_idx (int): The field index to read. For temperature, this is 4. 
        nside_out (int): The nside for the output map.
        cen_freq (float): The central frequency for the map.
    Returns:
        np.ndarray: The standard deviation map.
    """
    src_unit = fits_inspect.get_field_unit_str(fits_fn, field_idx[0], hdu=hdu)
    src_unit = convert_field_str_to_Unit(src_unit)

    source_skymap = hp.read_map(fits_fn, hdu=hdu, field=field_idx)

    m = change_variance_map_resolution(source_skymap, nside_out)
    m = np.sqrt(m) * (src_unit ** 0.5)

    m = m.to(u.K_CMB, equivalencies=u.cmb_equivalencies(cen_freq))

    logger.debug(f"physics_instrument_noise.planck_result_to_sd_map end")
    return m

def change_variance_map_resolution(m, nside_out):
    # For variance maps, because statistics
    power = 2

    # From PySM3 template.py's read_map function, with minimal alteration (added 'power'):
    m_dtype = fits_inspect.get_map_dtype(m)
    nside_in = hp.get_nside(m)
    if nside_out < nside_in:  # do downgrading in double precision
        m = hp.ud_grade(m.astype(np.float64), power=power, nside_out=nside_out)
    elif nside_out > nside_in:
        m = hp.ud_grade(m, power=power, nside_out=nside_out)
    m = m.astype(m_dtype, copy=False)
    # End of used portion
    return m

def get_sqrt_unit(src_unit):
    # Can't use PySM3's read_map() function because
    #     astropy.units will not parse "(K_CMB)^2" (I think)
    ok_units_k_cmb = ["(K_CMB)^2", "Kcmb^2"]
    ok_units_mjysr = ["(Mjy/sr)^2"]
    ok_units = [*ok_units_k_cmb, *ok_units_mjysr]
    if src_unit not in ok_units:
        raise ValueError(f"Wrong unit found in fits file. Found {src_unit}, expected one of {ok_units}.")
    if src_unit in ok_units_k_cmb:
        sqrt_unit = "K_CMB"
    else:
        sqrt_unit = "MJy/sr"
    return sqrt_unit
