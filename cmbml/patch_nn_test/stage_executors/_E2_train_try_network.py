import logging

from tqdm import tqdm

# import multiprocessing as mp
import torch
from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import LambdaLR
from omegaconf import DictConfig

import healpy as hp

from cmbml.core import Split, Asset
from cmbml.core.executor_base import BaseStageExecutor
from cmbml.core.asset_handlers import Config, PyTorchModel, HealpyMap, NumpyMap

# from cmbml.core.asset_handlers.asset_handlers_base import Config
# from cmbml.core.asset_handlers.pytorch_model_handler import PyTorchModel # Import for typing hint
# from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap
# from cmbml.core.asset_handlers.handler_npymap import NumpyMap
# from core.pytorch_dataset import TrainCMBMapDataset
from cmbml.patch_nn_test.dataset import TrainCMBMap2PatchDataset
# from cmbml.core.pytorch_transform import TrainToTensor
# from cmbml.cmbnncs_local.preprocessing.scale_methods_factory import get_scale_class
# from cmbml.cmbnncs_local.preprocessing.transform_pixel_rearrange import sphere2rect
from cmbml.utils.planck_instrument import make_instrument, Instrument
from cmbml.patch_nn_test.utils.display_help import show_patch
from cmbml.patch_nn_test.dummy_model import SimpleUNetModel
from cmbml.patch_nn_test.stage_executors._pytorch_executor_base import BasePyTorchModelExecutor


logger = logging.getLogger(__name__)


class TrainingTryNetworkExecutor(BasePyTorchModelExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="train_no_preprocess")

        self.out_model: Asset = self.assets_out["model"]
        out_model_handler: PyTorchModel

        # self.in_cmb_asset: Asset = self.assets_in["cmb_map"]
        # self.in_obs_assets: Asset = self.assets_in["obs_maps"]
        # self.in_norm_file: Asset = self.assets_in["norm_file"]
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
