import logging

from omegaconf import DictConfig

from cmbml.core import Split, Asset
from cmbml.core.executor_base import BaseStageExecutor
from cmbml.core.asset_handlers.pytorch_model_handler import PyTorchModel # Import for typing hint
from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap
from cmbml.utils.patch_healpix import make_pixel_index_lut


logger = logging.getLogger(__name__)


class MakeLutExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="make_lut")

        out_model_handler: PyTorchModel
        self.out_lut: Asset = self.assets_out["lut"]
        out_lut_handler: HealpyMap
        self.nside = cfg.scenario.nside
        self.nside_patch = cfg.model.patches.nside_patch

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

        patch_lut = make_pixel_index_lut(nside_obs=self.nside, nside_patches=self.nside_patch)
        self.out_lut.write(data=patch_lut)
