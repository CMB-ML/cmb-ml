from typing import Dict, Any
from pathlib import Path
import logging
import inspect
import camb

import numpy as np
import healpy as hp


# Based on https://camb.readthedocs.io/en/latest/CAMBdemo.html


logger = logging.getLogger(__name__)


def make_cmb_ps(cosmo_params, lmax, cmb_ps_fp: Path) -> None:
    #Set up a new set of parameters for CAMB
    pars = setup_camb(cosmo_params, lmax)
    results = run_camb(pars)
    results.save_cmb_power_spectra(filename=cmb_ps_fp)
    return


def setup_camb(cosmo_params: Dict[str, Any], lmax:int) -> camb.CAMBparams:
    pars = camb.CAMBparams()

    # CAMB sets what I naively see as two different 
    def get_camb_input_params(method):
        sig = inspect.signature(method)
        return [param.name for param in sig.parameters.values() if param.name != 'self']

    set_cosmology_params = get_camb_input_params(pars.set_cosmology)
    init_power_params = get_camb_input_params(pars.InitPower.set_params)

    # Split the cosmo_params dictionary
    set_cosmology_args = {k: v for k, v in cosmo_params.items() if k in set_cosmology_params}
    init_power_args = {k: v for k, v in cosmo_params.items() if k in init_power_params}

    # Set the parameters
    pars.set_cosmology(**set_cosmology_args)
    pars.InitPower.set_params(**init_power_args)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    return pars


def run_camb(pars: camb.CAMBparams) -> camb.CAMBdata:
    results = camb.get_results(pars)
    return results


def map2ps(skymap: np.ndarray, lmax: int) -> np.ndarray:
    # Convert a map to a power spectrum using healpy; alternatives abound.
    ps = hp.anafast(skymap, lmax=lmax)
    ps = ps.T
    return ps


def convert_to_log_power_spectrum(ps_cl):
    # Converts from "Cl" to "Dl", also called the "Log Power Spectrum"
    ell = np.atleast_2d(np.arange(start=0, stop=ps_cl.shape[0])).T
    factor = ell * (ell + 1) / (2 * np.pi)
    ps_dl = ps_cl * factor
    return ps_dl
