from typing import Dict, Any
import logging
from pathlib import Path
from pysm3 import CMBLensed
import camb
from omegaconf.errors import ConfigAttributeError

from namer_dataset_output import SimFilesNamer
from physics_cmb import make_cmb_ps, map2ps, convert_to_log_power_spectrum

logger = logging.getLogger(__name__)


class CMBFactory:
    def __init__(self, conf, make_ps_if_absent=None):
        self.nside = conf.simulation.nside
        self.max_ell_for_camb = conf.simulation.cmb.ell_max
        self.wmap_param_labels = conf.simulation.cmb.wmap_params
        self.camb_param_labels = conf.simulation.cmb.camb_params_equiv
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

    def make_cmb_lensed(self, seed, sim_files: SimFilesNamer) -> CMBLensed:
        # Get the ps path only after you know it exists
        cmb_ps_fid_path = self._get_or_make_ps_path(sim_files)
        return CMBLensed(nside=self.nside,
                         cmb_spectra=cmb_ps_fid_path,
                         cmb_seed=seed,
                         max_nside=self.max_nside_pysm_component,
                         apply_delens=self.apply_delens,
                         delensing_ells=self.delensing_ells,
                         map_dist=self.map_dist)
    
    def _get_or_make_ps_path(self, sim_files: SimFilesNamer) -> Path:
        # returns the path for the power spectrum, but first,
        #    creates a power spectrum file if it does not exist
        path = sim_files.cmb_ps_fid_path
        if not path.exists():
            if self.make_ps_if_absent:
                self.make_ps(sim_files)
            else:
                raise FileNotFoundError(f"Cannot find {path}.")
        return path
    
    def make_ps(self, sim_files: SimFilesNamer) -> None:
        logger.debug(f"Making power spectrum for {sim_files.name}.")
        cosmo_params = sim_files.read_wmap_params_file()
        out_path = sim_files.cmb_ps_fid_path

        cosmo_params = self._translate_params_keys(cosmo_params)

        make_cmb_ps(cosmo_params, self.max_ell_for_camb, out_path)

    def _param_translation_dict(self):
        translation = {}
        for i in range(len(self.wmap_param_labels)):
            translation[self.wmap_param_labels[i]] = self.camb_param_labels[i]
        return translation
    
    def _translate_params_keys(self, src_params):
        translation_dict = self._param_translation_dict()
        target_dict = {}
        for k in src_params:
            if k == "chain_idx":
                continue
            target_dict[translation_dict[k]] = src_params[k]
        return target_dict


def make_cmb_maker(conf) -> CMBFactory:
    cmb_maker = CMBFactory(conf)
    return cmb_maker


def save_fid_cmb_map(cmb: CMBLensed, sim: SimFilesNamer):
    fid_cmb_map = cmb.map
    logger.debug(f"Saving fiducial cmb_map for {sim.name}")
    sim.write_fid_map(fid_cmb_map)


def save_der_cmb_ps(cmb: CMBLensed, sim: SimFilesNamer, lmax: int):
    logger.debug(f"Getting derived cmb_ps for {sim.name}")
    fid_cmb_map = cmb.map
    ps_cl = map2ps(fid_cmb_map, lmax)
    ps_dl = convert_to_log_power_spectrum(ps_cl)
    logger.debug(f"Saving fiducial cmb_ps for {sim.name}")
    camb.results.save_cmb_power_array(sim.cmb_ps_der_path,
                                      array=ps_dl,
                                      labels="TT EE BB TE EB TB")
