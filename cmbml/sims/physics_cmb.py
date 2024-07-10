from typing import Dict, Any, List, Union
from pathlib import Path
import logging
import inspect
import camb
from astropy.units import Quantity

import numpy as np
import healpy as hp


# Based on https://camb.readthedocs.io/en/latest/CAMBdemo.html


logger = logging.getLogger(__name__)


def make_camb_ps(cosmo_params, lmax) -> camb.CAMBdata:
    #Set up a new set of parameters for CAMB
    # logger.debug(f"Beginning CAMB")
    pars: camb.CAMBparams = setup_camb(cosmo_params, lmax)
    results: camb.CAMBdata = camb.get_results(pars)
    return results


def setup_camb(cosmo_params: Dict[str, Any], lmax:int) -> camb.CAMBparams:
    pars = camb.CAMBparams()

    set_cosmology_args, init_power_args = _split_cosmo_params_dict(cosmo_params, pars)

    pars.set_cosmology(**set_cosmology_args)
    pars.InitPower.set_params(**init_power_args)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    return pars


def _split_cosmo_params_dict(cosmo_params: Dict, camb_pars):
    # Turn cosmo_params (a single dictionary) into input suitable for CAMB
    def get_camb_input_params(method):
        sig = inspect.signature(method)
        return [param.name for param in sig.parameters.values() if param.name != 'self']

    set_cosmology_params = get_camb_input_params(camb_pars.set_cosmology)
    init_power_params = get_camb_input_params(camb_pars.InitPower.set_params)

    # Split the cosmo_params dictionary
    set_cosmo_args = {k: v for k, v in cosmo_params.items() if k in set_cosmology_params}
    init_power_args = {k: v for k, v in cosmo_params.items() if k in init_power_params}

    _ensure_all_params_used(set_cosmo_args, init_power_args, cosmo_params)
    # _log_camb_args(set_cosmo_args, init_power_args)

    return set_cosmo_args, init_power_args


def _ensure_all_params_used(set_cosmo_args, init_power_args, cosmo_params) -> None:
    out_params = list(set_cosmo_args.keys())
    out_params.extend(init_power_args.keys())
    for in_param in cosmo_params.keys():
        if in_param == "chain_idx":
            continue
        try:
            assert in_param in out_params
        except AssertionError:
            logger.warning(f"Parameter {in_param} not found in {out_params}.")


def _log_camb_args(set_cosmo_args, init_power_args) -> None:
    logger.debug(f"CAMB cosmology args: {set_cosmo_args}")
    logger.debug(f"CAMB init_power args: {init_power_args}")


def alm_downgrade(map_in, nside_out, lmax_in, lmax_out):
    """
    Downgrades a map from one nside to another, in the spherical
    harmonic transform space.

    Args:
        map_in (np.array): The input map
        nside_out (int): The nside of the output map
        lmax_in (int): The maximum l value of the input map
        lmax_out (int): The maximum l value of the output map

    Returns:
        The output map
    """
    # Get a_lm's from map_in
    nside_in = hp.get_nside(map_in)
    if nside_out > nside_in:
        raise ValueError("Map resolution is lower than specified output resolution.")
    elif nside_out == nside_in:
        logger.warning("Input and output resolutions are the same. Returning input map.")
        return map_in
    if lmax_in is None:
        # TODO: I don't like the magic number here
        lmax_in = 3*nside_in - 1

    alm_in = hp.map2alm(map_in, lmax = lmax_in)

    alm_out = hp.almxfl(alm_in, np.ones(lmax_out + 1))
    map_out = hp.alm2map(alm_out, nside_out)
    return map_out


def downgrade_map(cmb_maps: Union[np.ndarray, List[np.ndarray]], 
                  nside_out: int, 
                  lmax_in: int, 
                  lmax_out: int):
    """
    \doc
    Downgrades maps, preserving power spectrum.

    Args:
        cmb_maps (np.array): Input map or maps; map is shape (npix, ), maps is shape (n_maps, npix)
        nside_out (int): The nside of the output map
        lmax_in (int): The maximum l value of the input map
        lmax_out (int): The maximum l value of the output map

    Returns:
        The output map or maps (np.array or list of np.array)
    """
    try:
        # Assume single map; if not, cmb_maps.dtype will fail duck typing
        scaled_map = alm_downgrade(cmb_maps, nside_out=nside_out, lmax_in=lmax_in, lmax_out=lmax_out)
        # OLD  # scaled_map = hp.ud_grade(cmb_maps, nside_out=nside_out, dtype=cmb_maps.dtype)
    # except AttributeError:
    #     # healpy's ud_grade raises this for lists of maps
    #     scaled_map = [hp.ud_grade(cmb_map, 
    #                               nside_out=nside_out, 
    #                               dtype=cmb_map.dtype
    #                               ) for cmb_map in cmb_maps]
    except ValueError:
        # healpy's almxfl raises this for lists of maps
        scaled_map = [alm_downgrade(cmb_map, 
                                  nside_out=nside_out, 
                                  lmax_in=lmax_in, 
                                  lmax_out=lmax_out
                                  ) for cmb_map in cmb_maps]
    return scaled_map
