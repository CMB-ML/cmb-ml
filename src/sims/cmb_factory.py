import logging

from pathlib import Path
from pysm3 import CMBLensed
from omegaconf.errors import ConfigAttributeError
from ..core import Asset


logger = logging.getLogger(__name__)


class CMBFactory:
    def __init__(self, conf
                #  , make_ps_if_absent=None
                 ):
        self.nside = conf.simulation.nside_sky
        self.max_nside_pysm_component = None
        self.apply_delens = False
        self.delensing_ells = None
        self.map_dist = None

        # Remove?:
        # self.max_ell_for_camb = conf.simulation.cmb.ell_max
        # self.wmap_param_labels = conf.simulation.cmb.wmap_params
        # self.camb_param_labels = conf.simulation.cmb.camb_params_equiv
        # if make_ps_if_absent is None:
        #     try:
        #         self.make_ps_if_absent = conf.simulation.cmb.make_ps_if_absent
        #     except ConfigAttributeError as e:
        #         logger.error(f"CMBMaker needs either a make_ps_if_absent flag in the cmb " \
        #                      f"configuration yaml OR a make_ps_if absent argument to init.")
        #         logger.exception(e)
        #         raise e
    
    def make_cmb_lensed(self, seed, cmb_ps_fid_path) -> CMBLensed:
        return CMBLensed(nside=self.nside,
                         cmb_spectra=cmb_ps_fid_path,
                         cmb_seed=seed,
                         max_nside=self.max_nside_pysm_component,
                         apply_delens=self.apply_delens,
                         delensing_ells=self.delensing_ells,
                         map_dist=self.map_dist)
