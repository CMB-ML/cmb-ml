from typing import Dict, Any
from pathlib import Path
import logging
import inspect
import camb

import numpy as np
import healpy as hp


# Based on https://camb.readthedocs.io/en/latest/CAMBdemo.html


logger = logging.getLogger(__name__)


def make_camb_ps(cosmo_params, lmax) -> camb.CAMBdata:
    #Set up a new set of parameters for CAMB
    logger.debug(f"Beginning CAMB")
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
    _log_camb_args(set_cosmo_args, init_power_args)

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


# def map2ps(skymap: np.ndarray, lmax: int) -> np.ndarray:
#     ps = hp.anafast(skymap, lmax=lmax)
#     ps = ps.T
#     return ps


# def convert_to_log_power_spectrum(ps_cl: np.ndarray):
#     # Converts from "Cl" to "Dl", also called the "Log Power Spectrum"
#     ell = np.atleast_2d(np.arange(start=0, stop=ps_cl.shape[0])).T
#     factor = ell * (ell + 1) / (2 * np.pi)
#     ps_dl = ps_cl * factor
#     return ps_dl


def change_nside_of_map(cmb_map: np.ndarray, nside_out: int):
    scaled_map = hp.ud_grade(cmb_map, nside_out=nside_out, dtype=cmb_map.dtype)
    return scaled_map
