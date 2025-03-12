from typing import List, Dict, Callable
import logging

import numpy as np
import healpy as hp
import torch
from torch.utils.data import DataLoader

from omegaconf import DictConfig

from cmbml.core import (
    BaseStageExecutor, 
    Split,
    )


from cmbml.deepsphere_unet.deepsphere.unet_model import SphericalUNet

from cmbml.deepsphere_unet.deepsphere.bayesian_unet import BayesianSphericalUNet

from cmbml.core.asset_handlers.pytorch_model_handler import PyTorchModel  # Must be imported to get it registered
from cmbml.utils import make_instrument, Instrument, Detector


logger = logging.getLogger(__name__)


class BasePyTorchModelExecutor(BaseStageExecutor):
    dtype_mapping = {
        "float": torch.float32,
        "double": torch.float64
    }

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

    # TODO: Remove this? It's replaced in children classes
    def make_model(self):
        raise NotImplementedError("This method must be implemented in a child class.")
    #     logger.debug(f"Using {self.device} device")
    #     model = self.make_model(self.cfg)
    #     # logger.info(model)
    #     return model

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


class BaseDeepSphereModelExecutor(BasePyTorchModelExecutor):
    def __init__(self, cfg: DictConfig, stage_str) -> None:
        super().__init__(cfg, stage_str)
        
        # self.nside = cfg.scenario.nside
        npix = hp.nside2npix(self.nside)

        input_channels = cfg.scenario.detector_freqs 
        depth = cfg.model.deepsphere.network.depth 
        laplacian_type = cfg.model.deepsphere.network.laplacian_type
        kernel_size = cfg.model.deepsphere.network.kernel_size

        # input_c = len(input_channels) # Ideally should work, but hardcoded to 9
        input_c = 9

        self.model_dict = dict(N=npix, depth=depth, laplacian_type=laplacian_type, kernel_size=kernel_size)

    def try_model(self, model):
        dummy_input = torch.rand(1, hp.nside2npix(self.nside), 9, device=self.device)
        result = model(dummy_input)
        logger.debug(f"Output result size: {result.size()}")

    def make_model(self):
        return SphericalUNet(**self.model_dict)

class BayesianDeepSphereModelExecutor(BasePyTorchModelExecutor):
    def __init__(self, cfg: DictConfig, stage_str) -> None:
        super().__init__(cfg, stage_str)
        
        nside = cfg.scenario.nside
        npix = hp.nside2npix(nside)

        input_channels = cfg.scenario.detector_freqs 
        depth = cfg.model.deepsphere.network.depth 
        laplacian_type = cfg.model.deepsphere.network.laplacian_type
        kernel_size = cfg.model.deepsphere.network.kernel_size
        tau = cfg.model.deepsphere.network.tau
        length_scale = cfg.model.deepsphere.network.length_scale

        if cfg.splits.name == "1-1" or cfg.splits.name == "1-10":
            train_length = cfg.splits.Test.n_sims
        train_length = cfg.splits.Train.n_sims

        
        weight_regularizer = length_scale**2 / (tau * train_length)
        dropout_regularizer = 2 / (tau * train_length)
        # input_c = len(input_channels) # Ideally should work, but hardcoded to 9
        input_c = 9

        self.model_dict = dict(N=npix, depth=depth, laplacian_type=laplacian_type, kernel_size=kernel_size, weight_regularizer=weight_regularizer, dropout_regularizer=dropout_regularizer)

    def try_model(self, model):
        dummy_input = torch.rand(1, hp.nside2npix(self.nside), 9, device=self.device)
        mu, sigma = model(dummy_input)

        # logger.debug(f"Output result size: {result.size()}")
        logger.debug(f"Output mu size: {mu.size()}")
        logger.debug(f"Output sigma size: {sigma.size()}")
    def make_model(self):
        return BayesianSphericalUNet(**self.model_dict)