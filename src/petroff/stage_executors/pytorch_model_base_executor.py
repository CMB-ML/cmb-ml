from typing import List, Dict
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

from omegaconf import DictConfig

from core import (BaseStageExecutor, Split)

# from ..dataset import CMBMapDataset
# from ..dummymodel import DummyNeuralNetwork
from ..deepsphere_model.model import PetroffNet
from core.asset_handlers.handler_model_pytorch import PyTorchModel  # Must be imported for registration
from src.utils import make_instrument, Instrument, Detector


logger = logging.getLogger(__name__)

class BasePyTorchModelExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig, stage_str) -> None:
        super().__init__(cfg, stage_str)
        self.instrument: Instrument = make_instrument(cfg=cfg)

        self.n_dets = len(self.instrument.dets)
        self.nside = cfg.scenario.nside

    def choose_device(self, force_device=None) -> None:
        if force_device:
            self.device = force_device
        else:
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )

    def make_fn_template(self, split: Split, asset):
        context = dict(
            split=split.name,
            sim=self.name_tracker.sim_name_template,
            freq="{freq}"
        )
        with self.name_tracker.set_contexts(contexts_dict=context):
        # with self.name_tracker.set_context("split", split.name):
        #     # The following set_context is a bit hacky; we feed the template into itself so it is unchanged
        #     with self.name_tracker.set_context("sim", self.name_tracker.sim_name_template):

            this_path_pattern = str(asset.path)
        return this_path_pattern

    def match_data_precision(self, tensor):
        # TODO: Revisit
        # data_precision is the precision with which the data is written to file
        # model_precision is the precision with which the model is created
        # tensor is the loaded data
        # If the tensor precision doesn't match the models, convert it
        # If the tensor precision doesn't match data_precision... is there an issue?
        if self.model_precision == "float" and tensor.dtype is torch.float64:
            return tensor.float()
        if self.model_precision == "float" and tensor.dtype is torch.float32:
            return tensor
        else:
            message = f"BasePyTorchModelExecutor data conversion is partially implemented. Received from config model precision: {self.model_precision}, data precision: {self.data_precision}. Received a tensor with dtype: {tensor.dtype}."
            logger.error(message)
            raise NotImplementedError(message)

    def prep_data(self, tensor):
        return self.match_data_precision(tensor).to(self.device)

    def try_model(self, model):
        n_pix = (self.nside ** 2) * 12
        dummy_input = torch.rand(1, self.n_dets, n_pix, device=self.device)
        result = model(dummy_input)
        logger.info(f"Output result size: {result.size()}")


class PetroffModelExecutor(BasePyTorchModelExecutor):
    def __init__(self, cfg: DictConfig, stage_str) -> None:
        super().__init__(cfg, stage_str)
        self.model_precision = self.cfg.model.petroff.network.model_precision
        self.model_dict = dict(
            n_dets = self.n_dets,
            nside = self.nside,
            laplacian_type = self.cfg.model.petroff.network.laplacian_type,
            kernel_size = self.cfg.model.petroff.network.kernel_size,
            n_features = self.cfg.model.petroff.network.n_features
        )

    def make_model(self):
        logger.info(f"Using {self.device} device")
        model = PetroffNet(**self.model_dict).to(self.device)
        # logger.info(model)
        return model
