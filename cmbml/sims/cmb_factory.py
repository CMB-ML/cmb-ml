import logging

from pathlib import Path
from pysm3 import CMBLensed
from omegaconf.errors import ConfigAttributeError
from cmbml.core import Asset


logger = logging.getLogger(__name__)


class CMBFactory:
    def __init__(self, nside_sky):
        self.nside = nside_sky
        self.max_nside_pysm_component = None
        self.apply_delens = False
        self.delensing_ells = None
        self.map_dist = None

    def make_cmb_lensed(self, seed, cmb_ps_fid_path) -> CMBLensed:
        return CMBLensed(nside=self.nside,
                         cmb_spectra=cmb_ps_fid_path,
                         cmb_seed=seed,
                         max_nside=self.max_nside_pysm_component,
                         apply_delens=self.apply_delens,
                         delensing_ells=self.delensing_ells,
                         map_dist=self.map_dist)
