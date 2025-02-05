import logging

from tqdm import tqdm

import numpy as np
from torch.utils.data import DataLoader
from omegaconf import DictConfig

import healpy as hp

from cmbml.core import Split, Asset
from cmbml.core.asset_handlers import Config, PyTorchModel, HealpyMap, NumpyMap
from cmbml.demo_patch_nn.dataset import TestCMBPatchDataset
from cmbml.demo_patch_nn.stage_executors._pytorch_executor_base import BasePyTorchModelExecutor
from cmbml.demo_patch_nn.dummy_model import SimpleUNetModel


logger = logging.getLogger(__name__)


class PredictTryModelLoadExecutor(BasePyTorchModelExecutor):
    """
    Goal: Reassemble a map from patches.

    Use a dataset to iterate through patches (instead of simulations).
    """
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="predict")

        # self.out_cmb_asset: Asset = self.assets_out["cmb_map"]
        # out_cmb_handler: NumpyMap

        # self.in_obs_assets: Asset = self.assets_in["obs_maps"]
        # self.in_lut_asset: Asset = self.assets_in["lut"]
        # self.in_dataset_stats: Asset = self.assets_in["dataset_stats"]
        self.in_model_asset: Asset = self.assets_in["model"]
        # in_obs_map_handler: HealpyMap
        # in_lut_handler: NumpyMap
        # in_dataset_stats_handler: Config
        in_model_handler: PyTorchModel

        self.choose_device(cfg.model.patch_nn.test.device)
        # self.batch_size = cfg.model.patch_nn.test.batch_size
        # self.lut = self.in_lut_asset.read()

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

        model = SimpleUNetModel(
                           n_in_channels=len(self.instrument.dets),
                           note="Test case network"
                           )

        for model_epoch in self.model_epochs:
            with self.name_tracker.set_context("epoch", model_epoch):
                self.in_model_asset.read(model=model, epoch=model_epoch)
            model.eval().to(self.device)
            logger.info(f'Model at epoch "{model_epoch}" loaded successfully.')
