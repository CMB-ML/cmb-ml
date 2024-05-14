from typing import List, Dict
import logging

import numpy as np
from tqdm import tqdm
import healpy as hp

from omegaconf import DictConfig

from core import (
    BaseStageExecutor, 
    Split,
    Asset
    )
from ..make_ps import get_power as _get_power
from core.asset_handlers.psmaker_handler import NumpyPowerSpectrum
from core.asset_handlers import HealpyMap
from utils.physics_mask import downgrade_mask


logger = logging.getLogger(__name__)


class MakePredPowerSpectrumExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="make_pred_ps")

        self.out_auto_real: Asset = self.assets_out.get("auto_real", None)
        self.out_auto_pred: Asset = self.assets_out.get("auto_pred", None)
        self.out_x_real_pred: Asset = self.assets_out.get("x_real_pred", None)
        self.out_diff_real_pred: Asset = self.assets_out.get("diff_real_pred", None)
        self.out_abs_diff_real_pred: Asset = self.assets_out.get("abs_diff_real_pred", None)
        out_ps_handler: NumpyPowerSpectrum

        self.in_cmb_map_true: Asset = self.assets_in["cmb_map_real"]
        self.in_cmb_map_pred: Asset = self.assets_in["cmb_map_post"]
        in_cmb_map_handler: HealpyMap

        self.mask = self.get_mask()

        self.map_fields = self.cfg.scenario.map_fields
        self.use_pixel_weights = False

    def get_mask(self):
        mask_asset: Asset = self.assets_in.get("mask", None)
        mask_data = None
        if mask_asset:
            nside_out=self.cfg.scenario.nside
            with self.name_tracker.set_context("src_root", self.cfg.local_system.assets_dir):
                mask_data = self.in_mask.read(map_fields=mask_asset.use_fields)
            mask_data = downgrade_mask(mask_data, nside_out)
        return mask_data

    def get_power(self, mapp1, mapp2=None):
        return _get_power(mapp1, 
                          mapp2, 
                          mask=self.mask, 
                          use_pixel_weights=self.use_pixel_weights)

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute().")
        self.default_execute()

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
        auto_real_ps = self.get_power(true_map)
        self.out_auto_real.write(data=auto_real_ps)
        for epoch in self.model_epochs:
            with self.name_tracker.set_context("epoch", epoch):
                self.process_epoch(true_map)

    def process_epoch(self, true_map) -> None:
        pred_map = self.in_cmb_map_pred.read()
        auto_pred_ps = self.get_power(pred_map)
        self.out_auto_pred.write(data=auto_pred_ps)

        x_real_pred_ps = self.get_power(true_map, pred_map)
        self.out_x_real_pred.write(data=x_real_pred_ps)

        diff = true_map - pred_map
        diff_ps = self.get_power(diff)
        self.out_diff_real_pred.write(data=diff_ps)
        
        abs_diff = np.abs(diff)
        abs_diff_ps = self.get_power(abs_diff)
        self.out_abs_diff_real_pred.write(data=abs_diff_ps)
