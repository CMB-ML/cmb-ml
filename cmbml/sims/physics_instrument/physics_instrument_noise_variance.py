import logging

import numpy as np
import healpy as hp
from astropy.units import Unit
import pysm3.units as u
from astropy.cosmology import Planck15

import cmbml.utils.fits_inspection as fits_inspect


logger = logging.getLogger(__name__)


class VarianceNoise:
    do_cache = True
    # This class is used to generate noise maps from Planck's observation maps.
    # It is used in the NoiseCacheExecutor and SimCreatorExecutor classes.
    # This class is glue code. The functions afterwards are relevant to Physics.
    def __init__(self, cfg, name_tracker, asset_cache, asset_src=None):
        self.nside_out = cfg.scenario.nside
        self.name_tracker = name_tracker
        self.asset_noise_cache = asset_cache
        self.asset_noise_src = asset_src

    def convert_noise_var_map(self, fits_fn, hdu, field_idx, cen_freq):
        """
        Creates a standard deviation map from Planck's observation maps, which
        include covariance maps for the stokes parameters.

        We use onle the variance maps, which are II, QQ, and UU.
        """
        logger.debug(f"physics_instrument_noise_variance.planck_result_to_sd_map start")
        # try:
        res = planck_result_to_sd_map(self.nside_out, fits_fn, hdu, field_idx, cen_freq)
        # except <EXCEPTION_TYPE_HERE> as e:  # Exception expected when noise_src is None
        # <do stuff here>
        return res

    def get_noise_map(self, freq, field_str, noise_seed, center_frequency=None):
        # TODO: Why is center_frequency included?
        """
        Returns a noise map for the given frequency and field.
        Called externally in E_make_simulations.py

        Args:
            freq (int): The frequency of the map.
            field_str (str): The field of the map.
            noise_seed (int): The seed for the noise map.
            center_frequency (float): The center frequency of the detector.
        """
        with self.name_tracker.set_context('freq', freq):
            with self.name_tracker.set_context('field', field_str):
                sd_map = self.asset_noise_cache.read()
                noise_map = make_random_noise_map(sd_map, noise_seed, center_frequency)
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

def make_random_noise_map(sd_map, random_seed, center_frequency):
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
