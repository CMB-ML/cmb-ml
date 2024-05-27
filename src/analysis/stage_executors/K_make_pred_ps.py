from typing import List, Dict
import logging

from hydra.utils import instantiate
import numpy as np
from tqdm import tqdm
import healpy as hp

from omegaconf import DictConfig

from src.core import (
    BaseStageExecutor, 
    Split,
    Asset
    )
# from src.analysis.make_ps import get_power as _get_power
from src.core.asset_handlers.psmaker_handler import NumpyPowerSpectrum
from src.core.asset_handlers.healpy_map_handler import HealpyMap # Import for typing hint
from src.utils.physics_ps import get_auto_ps_result, get_x_ps_result, PowerSpectrum
from src.utils.physics_beam import NoBeam
from src.utils.physics_mask import downgrade_mask


logger = logging.getLogger(__name__)


class MakePredPowerSpectrumExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig, beam_type:str) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="make_pred_ps")

        self.out_auto_real: Asset = self.assets_out.get("auto_real", None)
        self.out_auto_pred: Asset = self.assets_out.get("auto_pred", None)
        # self.out_x_real_pred: Asset = self.assets_out.get("x_real_pred", None)
        out_ps_handler: NumpyPowerSpectrum

        self.in_cmb_map_true: Asset = self.assets_in["cmb_map_real"]
        self.in_cmb_map_pred: Asset = self.assets_in["cmb_map_post"]
        self.in_mask: Asset = self.assets_in.get("mask", None)
        in_cmb_map_handler: HealpyMap

        self.nside_out = self.cfg.scenario.nside
        self.mask_threshold = self.cfg.model.analysis.mask_threshold
        self.mask = None
        self.beam = cfg.model.analysis.get(beam_type, None)

        self.lmax = int(cfg.model.analysis.lmax_ratio * self.nside_out)
        self.use_pixel_weights = False

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute().")
        self.mask = self.get_mask()
        self.beam = self.get_beam()
        self.default_execute()

    def get_mask(self):
        mask = None
        if self.in_mask:
            with self.name_tracker.set_context("src_root", self.cfg.local_system.assets_dir):
                logger.info(f"Using mask from {self.in_mask.path}")
                mask = self.in_mask.read(map_fields=self.in_mask.use_fields)[0]
            mask = downgrade_mask(mask, self.nside_out, threshold=self.mask_threshold)
            # self.show_mask(mask)
        else:
            logger.warning("Not using any mask for calculating power spectra.")
        return mask

    # def show_mask(self, mask):
    #     """
    #     TODO: Move this elsewhere
    #     """
    #     import matplotlib.pyplot as plt
    #     hp.mollview(mask)
    #     plt.savefig("Mask.png")

    def get_beam(self):
        # Partially instantiate the beam object, defined in the hydra configs
        # Currently tested are GaussianBeam and NoBeam, which differ only in how they are instantiated
        beam = instantiate(self.beam)
        beam = beam(lmax=self.lmax)
        return beam

    def process_split(self, 
                      split: Split) -> None:
        logger.info(f"Running {self.__class__.__name__} process_split() for split: {split.name}.")
        for sim in tqdm(split.iter_sims()):
            with self.name_tracker.set_context("sim_num", sim):
                self.process_sim()

    def process_sim(self) -> None:
        true_map: np.ndarray = self.in_cmb_map_true.read()
        if true_map.shape[0] == 3 and self.map_fields == "I":
            true_map = true_map[0]
        auto_real_ps = get_auto_ps_result(true_map,
                                          mask=self.mask,
                                          lmax=self.lmax,
                                          beam=self.beam)
        ps = auto_real_ps.conv_dl
        ps2 = auto_real_ps.deconv_dl
        self.out_auto_real.write(data=ps)
        for epoch in self.model_epochs:
            with self.name_tracker.set_context("epoch", epoch):
                self.process_epoch(true_map)

    def process_epoch(self, true_map) -> None:
        pred_map = self.in_cmb_map_pred.read()
        auto_pred_ps = get_auto_ps_result(pred_map,
                                          mask=self.mask,
                                          lmax=self.lmax,
                                          beam=self.beam)
        ps = auto_pred_ps.conv_dl
        self.out_auto_pred.write(data=ps)

        # true_beam = NoBeam(lmax=self.lmax)
        # x_real_pred_ps = get_x_ps_result(map_data1=pred_map, 
        #                                  map_data2=true_map,
        #                                  mask_data=self.mask,
        #                                  beam1=self.beam,
        #                                  beam2=true_beam)
        # ps = x_real_pred_ps.conv_dl
        # self.out_x_real_pred.write(data=ps)


class PyILCMakePSExecutor(MakePredPowerSpectrumExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, "beam_pyilc")


class OtherMakePSExecutor(MakePredPowerSpectrumExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, "beam_other")
