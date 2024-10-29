import logging

import numpy as np
from astropy.units import Quantity, Unit
import healpy as hp
import pysm3.units as u
from astropy.cosmology import Planck15

import cmbml.utils.fits_inspection as fits_inspect
from cmbml.utils.fits_inspection import get_num_fields_in_hdr


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
        n_map_fields = get_num_fields_in_hdr(fits_fn=src_path, hdu=hdu)
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
        for field_str in detector.fields:
            field_idx = self.get_field_idx(src_path, field_str)
            st_dev_skymap = self.convert_noise_var_map(fits_fn=src_path,
                                                       hdu=hdu,
                                                       field_idx=field_idx,
                                                       cen_freq=detector.cen_freq)
            with self.name_tracker.set_contexts(dict(freq=freq, field=field_str)):
                self.write_wrapper(data=st_dev_skymap, field_str=field_str)

    def convert_noise_var_map(self, fits_fn, hdu, field_idx, cen_freq):
        """
        Creates a standard deviation map from Planck's observation maps, which
        include covariance maps for the stokes parameters.

        We use only the variance maps, which are II, QQ, and UU.
        """
        logger.debug(f"physics_instrument_noise_variance.planck_result_to_sd_map start")
        # try:
        res = planck_result_to_sd_map(self.nside_out, fits_fn, hdu, field_idx, cen_freq)
        # except <EXCEPTION_TYPE_HERE> as e:  # Exception expected when noise_src is None
        # <do stuff here>
        return res

    def write_wrapper(self, data: Quantity, field_str):
        """
        Wraps the write method for the noise cache asset to ensure proper column names and units.

        Parameters:
        data (Quantity): The standard deviation map to write to the noise cache.
        field_str (str): The field string (One of T,Q,U).
        """
        units = data.unit
        data = data.value
        
        # We want to give some indication that for I field, this is from the II covariance (or QQ, UU)
        col_name = field_str + field_str
        logger.debug(f'Writing NoiseCache map to path: {self.out_scale_cache.path}')
        self.out_scale_cache.write(data=data,
                                   column_names=[col_name],
                                   column_units=[units])
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
    Convert a Planck variance map to a standard deviation map.

    In the observation maps provided by Planck, fields 0,1,2 are stokes 
    parameters for T, Q, and U (resp). The HITMAP is field 3. The remainder
    are variance maps: II, IQ, IU, QQ, QU, UU. We use variance maps to generate
    noise maps, albeit with the huge simplification of ignoring covariance 
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
    source_skymap = hp.read_map(fits_fn, hdu=hdu, field=field_idx)

    m = change_variance_map_resolution(source_skymap, nside_out)
    m = np.sqrt(m)
    
    src_unit = fits_inspect.get_field_unit(fits_fn, hdu, field_idx)
    sqrt_unit = get_sqrt_unit(src_unit)

    # Convert MJy/sr to K_CMB (I think, TODO: Verify)
    # This is an oversimplification applied to the 545 and 857 GHz bands
    # something about "very sensitive to band shape" for sub-mm bands (forgotten source)
    # This may be a suitable first-order approximation
    if sqrt_unit == "MJy/sr":
        m = (m * u.MJy / u.sr).to(
            u.K, equivalencies=u.thermodynamic_temperature(cen_freq, Planck15.Tcmb0)
        ).value

    m = m * Unit(sqrt_unit)
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
