#TODO: REINSTATE USE OF THIS FILE

import logging

import numpy as np
import healpy as hp
import pysm3.units as u
from astropy.cosmology import Planck15

import utils.fits_inspection as fits_inspect


logger = logging.getLogger(__name__)


def planck_result_to_sd_map(fits_fn, hdu, field_idx, nside_out, cen_freq):
    logger.debug(f"physics_instrument_noise.planck_result_to_sd_map start")
    source_skymap = hp.read_map(fits_fn, hdu=hdu, field=field_idx)

    m = _change_variance_map_resolution(source_skymap, nside_out)
    m = np.sqrt(m)
    
    src_unit = fits_inspect.get_field_unit(fits_fn, hdu, field_idx)
    sqrt_unit = _get_sqrt_unit(src_unit)

    # Convert MJy/sr to K_CMB (I think, TODO: Verify)
    # This is an oversimplification applied to the 545 and 857 GHz bands
    # something about "very sensitive to band shape" for sub-mm bands (forgotten source)
    # This may be a suitable first-order approximation
    if sqrt_unit == "MJy/sr":
        m = (m * u.MJy / u.sr).to(
             u.K, equivalencies=u.thermodynamic_temperature(cen_freq, Planck15.Tcmb0)
        ).value

    logger.debug(f"physics_instrument_noise.planck_result_to_sd_map end")
    return m


def make_random_noise_map(sd_map, random_seed, center_frequency):
    #TODO: set units when redoing this function
    # logger.debug(f"physics_instrument_noise.make_random_noise_map start")
    rng = np.random.default_rng(random_seed)
    noise_map = rng.normal(scale=sd_map, size=sd_map.size)
    noise_map = u.Quantity(noise_map, u.K_CMB, copy=False)
    #TODO: take units as output instead of the following line.
    # noise_map = noise_map.to(u.uK_RJ, equivalencies=u.cmb_equivalencies(center_frequency))
    # logger.debug(f"physics_instrument_noise.make_random_noise_map end")
    return noise_map


def _change_variance_map_resolution(m, nside_out):
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


def _get_sqrt_unit(src_unit):
    # Can't use PySM3's read_map() function because
    #     astropy.units will not parse "(K_CMB)^2" (I think)
    ok_units_k_cmb = ["(K_CMB)^2", "Kcmb^2"]
    ok_units_mjysr = ["(Mjy/sr)^2"]
    ok_units = [*ok_units_k_cmb, *ok_units_mjysr]
    # TODO: Use logging
    # log.info(f"Units for map {self.ref_map_fn} are {unit}")
    if src_unit not in ok_units:
        raise ValueError(f"Wrong unit found in fits file. Found {src_unit}, expected one of {ok_units}.")
    if src_unit in ok_units_k_cmb:
        sqrt_unit = "K_CMB"
    else:
        sqrt_unit = "MJy/sr"
    return sqrt_unit
