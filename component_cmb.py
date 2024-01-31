from pathlib import Path
import logging
import inspect
from pysm3 import CMBLensed
import camb
from omegaconf.errors import ConfigAttributeError

import numpy as np
import healpy as hp

from hydra_filesets import SimFiles


# Based on https://camb.readthedocs.io/en/latest/CAMBdemo.html


logger = logging.getLogger(__name__)


def make_cmb_ps(cosmo_params, lmax, cmb_ps_fp: Path) -> None:
    #Set up a new set of parameters for CAMB
    pars = camb.CAMBparams()

    # I don't know what CAMB is doing, I just know I've got some lump of parameters to 
    #   plug in to two different functions, so I read signatures to split the params up
    #   this is a bit hacky, but is more flexible for now.
    set_cosmology_params = get_param_names(pars.set_cosmology)
    init_power_params = get_param_names(pars.InitPower.set_params)

    # Split the cosmo_params dictionary
    set_cosmology_args = {k: v for k, v in cosmo_params.items() if k in set_cosmology_params}
    init_power_args = {k: v for k, v in cosmo_params.items() if k in init_power_params}

    # Set the parameters
    pars.set_cosmology(**set_cosmology_args)
    pars.InitPower.set_params(r=0, **init_power_args)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    
    results = camb.get_results(pars)

    results.save_cmb_power_spectra(filename=cmb_ps_fp)
    return


def get_param_names(method):
    sig = inspect.signature(method)
    return [param.name for param in sig.parameters.values() if param.name != 'self']


class CMBMaker:
    def __init__(self, conf, make_ps_if_absent=None):
        self.nside = conf.simulation.nside
        self.max_ell_for_camb = conf.simulation.cmb.ell_max
        self.max_nside_pysm_component = None
        self.apply_delens = False
        self.delensing_ells = None
        self.map_dist = None
        if make_ps_if_absent is None:
            try:
                self.make_ps_if_absent = conf.simulation.cmb.make_ps_if_absent
            except ConfigAttributeError as e:
                logger.error(f"CMBMaker needs either a make_ps_if_absent flag in the cmb " \
                             f"configuration yaml OR a make_ps_if absent argument to init.")
                logger.exception(e)
                raise e

    def make_cmb_lensed(self, seed, sim_files: SimFiles) -> CMBLensed:
        # Get the ps path only after you know it exists
        cmb_ps_fid_path = self._get_or_make_ps_path(sim_files)
        return CMBLensed(nside=self.nside,
                         cmb_spectra=cmb_ps_fid_path,
                         cmb_seed=seed,
                         max_nside=self.max_nside_pysm_component,
                         apply_delens=self.apply_delens,
                         delensing_ells=self.delensing_ells,
                         map_dist=self.map_dist)
    
    def _get_or_make_ps_path(self, sim_files: SimFiles) -> Path:
        path = sim_files.cmb_ps_fid_path
        if not path.exists():
            if self.make_ps_if_absent:
                self.make_ps(sim_files)
            else:
                raise FileNotFoundError(f"Cannot find {path}.")
        return path
    
    def make_ps(self, sim_files: SimFiles) -> None:
        cosmo_params = sim_files.read_wmap_params_file()
        out_path = sim_files.cmb_ps_fid_path
        make_cmb_ps(cosmo_params, self.max_ell_for_camb, out_path)


def make_cmb_maker(conf) -> CMBMaker:
    # make whatever path keeper-tracker
    cmb_maker = CMBMaker(conf)
    return cmb_maker


def save_fid_cmb_map(cmb: CMBLensed, sim: SimFiles):
    fid_cmb_map = cmb.map
    sim.write_fid_map(fid_cmb_map)


def save_der_cmb_ps(cmb: CMBLensed, sim: SimFiles, lmax: int):
    fid_cmb_map = cmb.map
    ps = hp.anafast(fid_cmb_map, lmax=lmax)
    ps = ps.T
    ell = np.atleast_2d(np.arange(start=0, stop=ps.shape[0])).T
    factor = ell * (ell + 1) / (2 * np.pi)
    ps = ps * factor
    camb.results.save_cmb_power_array(sim.cmb_ps_der_path,
                                      array=ps,
                                      labels="TT EE BB TE EB TB")
    # hp.fitsfunc.write_cl(filename=sim.cmb_ps_der_path,
    #                      cl=ps,
    #                      overwrite=True)
