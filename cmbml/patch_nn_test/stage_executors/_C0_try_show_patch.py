# Goal: create Executor that loads and shows patches of a map, testing C_choose_patches.py

import logging

from cmbml.utils.planck_instrument import make_instrument, Instrument

from cmbml.core import BaseStageExecutor, Asset
from cmbml.core.asset_handlers.asset_handlers_base import PlainText
from cmbml.core.asset_handlers.qtable_handler import QTableHandler # Import to register handler
from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap # Import for typing hint
from cmbml.utils.patch_healpix import get_patch_pixels
from cmbml.patch_nn_test.utils.display_help import show_patch


logger = logging.getLogger(__name__)


class TryShowPatchExecutor(BaseStageExecutor):
    """
    info here
    """
    def __init__(self, cfg):
        super().__init__(cfg, stage_str='try_show_patch')

        self.in_patch_id: Asset = self.assets_in['patch_id']
        in_map_handler: HealpyMap
        self.in_cmb_map: Asset = self.assets_in['cmb_map']
        self.in_obs_maps: Asset = self.assets_in['obs_maps']
        in_patch_id_handler: PlainText
        in_det_table: Asset  = self.assets_in['planck_deltabandpass']
        in_det_table_handler: QTableHandler

        det_info = in_det_table.read()
        self.instrument: Instrument = make_instrument(cfg, det_info)

        self.nside_obs = cfg.scenario.nside
        self.nside_patch = cfg.model.patches.nside_patch

        self.show_n = cfg.pipeline[self.stage_str].override_n_sims

    def execute(self):
        # TODO: Implement support for other map fields
        if self.map_fields != "I":
            raise NotImplementedError("This executor only supports I maps.")
        self.default_execute()

    def process_split(self, split):
        for sim_num in range(split.n_sims):  # break at end of loop; observe just the first sim
            obs_maps = []
            with self.name_tracker.set_context("sim_num", sim_num):
                patch_id = self.in_patch_id.read(astype=int)
                if self.map_fields != "I":
                    raise NotImplementedError("This executor only supports I maps.")
                cmb_map = self.in_cmb_map.read()[0]  # Hard-lock to temperature map
                for freq, det in self.instrument.dets.items():
                    with self.name_tracker.set_context("freq", freq):
                        if self.map_fields != "I":
                            raise NotImplementedError("This executor only supports I maps.")
                        obs_maps.append(self.in_obs_maps.read()[0])  # Hard-lock to temperature map
            patch_pixels = get_patch_pixels(patch_id, self.nside_patch, self.nside_obs)
            title = f"Simulation {split.name} {sim_num:04d} Patch {patch_id}"
            
            cmb_patch = cmb_map[patch_pixels].value
            obs_patches = [obs_map[patch_pixels].value for obs_map in obs_maps]

            show_patch(cmb_patch, obs_patches, title)
            if sim_num >= self.show_n - 1:
                break
