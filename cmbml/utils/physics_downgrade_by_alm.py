import logging

import numpy as np
import healpy as hp
import pysm3.units as u
import pysm3


logger = logging.getLogger(__name__)


def downgrade_noise_by_alm(some_map, target_nside):
    if hp.get_nside(some_map) == target_nside:
        logger.info("The map is already at the target nside.")
        return some_map
    try:
        map_unit = some_map.unit
    except AttributeError:
        map_unit = None
    source_nside = hp.get_nside(some_map)
    assert target_nside <= source_nside/2, "Target nside must be less than the source nside"
    lmax_source = 3 * source_nside - 1
    alm = hp.map2alm(some_map, lmax=lmax_source)

    lmax_target = int(3 * target_nside - 1)
    alm_filter = np.zeros(lmax_source+1)
    alm_filter[:lmax_target+1] = 1
    alm_filtered = hp.almxfl(alm, alm_filter)
    some_map_filtered = hp.alm2map(alm_filtered, nside=target_nside)
    if map_unit is not None:
        some_map_filtered = u.Quantity(some_map_filtered, unit=map_unit)
    return some_map_filtered


def pysm3_downgrade_T_by_alm(source_map, nside_out, src_beam_fwhm, tgt_beam_fwhm):
    """
    The PySM3 function to downgrade a temperature map by smoothing it with a 
    beam window function assumes that the map will include polarization.
    
    I want to be able to deconvolve and reconvolve without multiple SHTs.

    Parameters
    ----------
    source_map : np.ndarray
        The input temperature map to be downgraded.
    nside_out : int
        The desired output nside for the downgraded map.
    src_beam_fwhm : u.Quantity
        The full width at half maximum (FWHM) of the source beam, in units
        compatible with astropy.units (e.g., u.arcmin).
    tgt_beam_fwhm : u.Quantity
        The full width at half maximum (FWHM) of the target beam, in units
        compatible with astropy.units (e.g., u.arcmin).
    """
    nside_src = hp.get_nside(source_map)
    if nside_src == nside_out:
        lmax = int(2.5 * nside_out)
    elif nside_out > nside_src:
        lmax = int(2.5 * nside_src)
    elif nside_out < nside_src:
        lmax = int(1.5 * nside_src)
    logger.info("Setting lmax to %d", lmax)

    src_beam_size_rad = src_beam_fwhm.to(u.rad).value
    src_beam = hp.gauss_beam(fwhm=src_beam_size_rad, lmax=lmax)
    
    tgt_beam_size_rad = tgt_beam_fwhm.to(u.rad).value
    tgt_beam = hp.gauss_beam(fwhm=tgt_beam_size_rad, lmax=lmax)

    beam_ratio = tgt_beam / src_beam

    # Another bug in PySM3; it supports beam window only for maps that include
    #    polarization. Instead, I just duplicate relevant portion of their code here.
    # smoothed_map = pysm3.apply_smoothing_and_coord_transform(
    #     input_map=source_map,
    #     fwhm=None,  # Not used when beam_window is provided
    #     beam_window=beam_ratio,
    #     output_nside=self.nside_out,
    #     lmax=self.lmax,
    #     return_healpix=True
    # )

    alm = pysm3.map2alm(input_map=source_map,
                        nside=nside_out,
                        lmax=lmax,
                        map2alm_lsq_maxiter=0)
    hp.smoothalm(alm, beam_window=beam_ratio, inplace=True)
    smoothed_map = hp.alm2map(alm, nside=nside_out, pixwin=False)
    return smoothed_map
