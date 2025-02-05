import logging

import torch
from omegaconf import DictConfig

from cmbml.core import Asset
from cmbml.core.asset_handlers import Config, PyTorchModel, HealpyMap

from cmbml.demo_patch_nn.dummy_model import SimpleUNetModel
from cmbml.demo_patch_nn.stage_executors._pytorch_executor_base import BasePyTorchModelExecutor


logger = logging.getLogger(__name__)


class TrainingTryNetworkExecutor(BasePyTorchModelExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="train_no_preprocess")

        self.out_model: Asset = self.assets_out["model"]
        out_model_handler: PyTorchModel

        # self.in_cmb_asset: Asset = self.assets_in["cmb_map"]
        # self.in_obs_assets: Asset = self.assets_in["obs_maps"]
        # self.in_dataset_stats: Asset = self.assets_in["dataset_stats"]
        self.in_model: Asset = self.assets_in["model"]
        # self.in_all_p_ids_asset: Asset = self.assets_in["patch_dict"]
        in_norm_handler: Config
        in_cmb_map_handler: HealpyMap
        in_obs_map_handler: HealpyMap
        in_model_handler: PyTorchModel
        in_all_p_ids_handler: Config

        # self.nside_patch = cfg.model.patches.nside_patch

        self.choose_device(cfg.model.patch_nn.train.device)
        # self.n_epochs   = cfg.model.patch_nn.train.n_epochs
        # self.batch_size = cfg.model.patch_nn.train.batch_size

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

        model = self.make_model().to(self.device)

        print(model)

        input_ex = torch.randn(4, 9, 128, 128).to(self.device)
        output = model(input_ex)
        print(output.size())

        exit()

    def make_model(self):
        model = SimpleUNetModel(
                           n_in_channels=len(self.instrument.dets),
                           note="Test case network"
                           )
        return model
