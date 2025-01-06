import logging

import torch
from omegaconf import DictConfig

from cmbml.core.executor_base import BaseStageExecutor
from cmbml.core.asset_handlers.asset_handlers_base import Config
from cmbml.utils.planck_instrument import make_instrument, Instrument
from cmbml.core import Split, Asset


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
