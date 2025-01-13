# Goal: Create an Executor that loads all training patch IDs and shows a 
#       heatmap of the distribution

import logging

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from cmbml.core import BaseStageExecutor, Split, Asset

from cmbml.core import BaseStageExecutor, Asset
from cmbml.core.asset_handlers.asset_handlers_base import PlainText
from cmbml.utils.patch_healpix import get_inverse_nside


logger = logging.getLogger(__name__)


class TryShowPatchDistExecutor(BaseStageExecutor):
    """
    info here
    """
    def __init__(self, cfg):
        super().__init__(cfg, stage_str='try_show_patch_dist')

        self.in_patch_id: Asset = self.assets_in['patch_id']
        in_patch_id_handler: PlainText

        self.nside_obs = cfg.scenario.nside
        self.nside_patch = cfg.model.patches.nside_patch

    def execute(self):
        self.default_execute()

    def process_split(self, split: Split):
        used_patches = []
        for i in range(split.n_sims):
            with self.name_tracker.set_context("sim_num", i):
                patch_id = self.in_patch_id.read()
                used_patches.append(int(patch_id))
        map_nside = get_inverse_nside(nside_obs=self.nside_obs, 
                                      nside_patches=self.nside_patch)
        m = np.zeros(hp.nside2npix(map_nside))
        for patch in used_patches:
            m[patch] += 1
        m = hp.ma(m)
        m.mask = m == 0
        hp.mollview(m, title='Patch Distribution')
        plt.show()
        plt.hist(m)
        plt.show()

