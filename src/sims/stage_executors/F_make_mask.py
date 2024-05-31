"""
We need to make a mask for the power spectrum analysis.
This is a simple task, but it is important to ensure consistent results.
"""
import logging

from omegaconf import DictConfig

from src.core import BaseStageExecutor, Asset
from src.core.asset_handlers.healpy_map_handler import HealpyMap # Import for typing hint
from src.utils.physics_mask import downgrade_mask


logger = logging.getLogger(__name__)


class MaskCreatorExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following stage_str must match the pipeline yaml
        super().__init__(cfg, stage_str='make_mask')

        self.out_mask: Asset = self.assets_out['mask']
        out_mask_handler: HealpyMap

        self.in_mask: Asset = self.assets_in['mask']
        in_mask_handler: HealpyMap

        self.nside_out = cfg.scenario.nside
        self.mask_threshold = self.cfg.model.analysis.mask_threshold

    def execute(self) -> None:
        mask = self.get_mask()
        mask = downgrade_mask(mask, self.nside_out, threshold=self.mask_threshold)
        self.out_mask.write(data=mask)

    def get_mask(self):
        with self.name_tracker.set_context("src_root", self.cfg.local_system.assets_dir):
            logger.info(f"Using mask from {self.in_mask.path}")
            mask = self.in_mask.read(map_fields=self.in_mask.use_fields)[0]
            return mask
