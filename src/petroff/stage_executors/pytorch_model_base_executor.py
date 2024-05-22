import logging

import torch

from omegaconf import DictConfig

from core import (BaseStageExecutor, Split)

from src.petroff.deepsphere_model.model import PetroffNet
from src.core.asset_handlers.pytorch_model_handler import PyTorchModel  # Must be imported for registration
from src.utils import make_instrument, Instrument
from src.core.asset_handlers.healpy_map_handler import HealpyMap



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


class PetroffModelExecutor(BasePyTorchModelExecutor):
    def __init__(self, cfg: DictConfig, stage_str) -> None:
        super().__init__(cfg, stage_str)
        self.model_precision = self.cfg.model.petroff.network.model_precision
        self.model_dict = dict(
            n_dets = self.n_dets,
            nside = self.nside,
            laplacian_type = self.cfg.model.petroff.network.laplacian_type,
            kernel_size = self.cfg.model.petroff.network.kernel_size,
            n_features = self.cfg.model.petroff.network.n_features,
            initialization = self.cfg.model.petroff.network.initialization
        )

    def make_model(self):
        logger.debug(f"Using {self.device} device")
        model = PetroffNet(**self.model_dict).to(self.device)
        # logger.info(model)
        return model

    def try_model(self, model):
        n_pix = (self.nside ** 2) * 12
        dummy_input = torch.rand(1, self.n_dets, n_pix, device=self.device)
        result = model(dummy_input)
        logger.info(f"Output result size: {result.size()}")
