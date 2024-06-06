from typing import List, Dict
import logging
from itertools import product

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
from src.core.asset_handlers.healpy_map_handler import HealpyMap # Import for typing hint
from src.utils.physics_mask import downgrade_mask


logger = logging.getLogger(__name__)


class CommonPostExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig, stage_str: str) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str)

        self.out_cmb_map_real: Asset = self.assets_out["cmb_map"]
        out_ps_handler: HealpyMap

        self.in_cmb_map: Asset = self.assets_in["cmb_map"]
        self.in_mask: Asset = self.assets_in.get("mask", None)
        in_cmb_map_handler: HealpyMap

        # Basic parameters
        self.nside_out = self.cfg.scenario.nside
        self.lmax = int(cfg.model.analysis.lmax_ratio * self.nside_out)

        # Prepare to load mask (in execute())
        self.mask_threshold = self.cfg.model.analysis.mask_threshold

        self.use_pixel_weights = False

        # Prepare to load beam and mask in execute()
        self.beam = None
        self.mask = None

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute().")
        self.mask = self.get_mask()
        self.beam = self.get_beam()
        self.default_execute()

    def get_mask(self):
        mask = None
        with self.name_tracker.set_context("src_root", self.cfg.local_system.assets_dir):
            logger.info(f"Using mask from {self.in_mask.path}")
            mask = self.in_mask.read(map_fields=self.in_mask.use_fields)[0]
        mask = downgrade_mask(mask, self.nside_out, threshold=self.mask_threshold)
        return mask

    def get_beam(self):
        # Partially instantiate the beam object, defined in the hydra configs
        beam = instantiate(self.beam_cfg)
        beam = beam(lmax=self.lmax)
        return beam

    def process_split(self, 
                      split: Split) -> None:
        logger.info(f"Running {self.__class__.__name__} process_split() for split: {split.name}.")
        
        epochs = self.model_epochs if self.model_epochs else [""]

        for epoch in epochs:
            for sim in tqdm(split.iter_sims()):
                context_params = dict(epoch=epoch, sim_num=sim)
                with self.name_tracker.set_contexts(context_params):
                    self.process_sim()

    def process_sim(self) -> None:
        # Get power spectrum for realization
        cmb_map: np.ndarray = self.in_cmb_map.read()
        if cmb_map.shape[0] == 3 and self.map_fields == "I":
            cmb_map = cmb_map[0]

        # Apply the mask
        post_map = hp.ma(cmb_map)
        post_map.mask = np.logical_not(self.mask)

        # Deconvolve the beam
        post_map = self.deconv(post_map)
        post_map = hp.ma(post_map)
        post_map.mask = np.logical_not(self.mask)

        # Remove the dipole and monopole
        post_map = hp.remove_dipole(post_map)
        self.out_cmb_map_real.write(data=post_map)

    def deconv(self, data) -> np.ndarray:
        # Convert to spherical harmonic space (a_lm)
        alm_in = hp.map2alm(data, lmax=self.lmax)
        
        # Deconvolve the beam
        alm_deconv = hp.almxfl(alm_in, 1 / self.beam.beam[:self.lmax])

        # Convert back to map space
        map_deconv = hp.alm2map(alm_deconv, nside=self.nside_out)

        return map_deconv


class CommonRealPostExecutor(CommonPostExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="common_post_map_real")
        self.out_cmb_map: Asset = self.assets_out["cmb_map"]
        self.in_cmb_map: Asset = self.assets_in["cmb_map"]
        self.beam_cfg = cfg.model.analysis.beam_real

    def deconv(self, data):
        """
        No-op because the realization map was never convolved
        """
        return data


class CommonCMBNNCSPredPostExecutor(CommonPostExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="common_post_map_pred")
        self.out_cmb_map: Asset = self.assets_out["cmb_map"]
        self.in_cmb_map: Asset = self.assets_in["cmb_map"]
        self.beam_cfg = cfg.model.analysis.beam_cmbnncs


class CommonPyILCPredPostExecutor(CommonPostExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="common_post_map_pred")
        self.out_cmb_map: Asset = self.assets_out["cmb_map"]
        self.in_cmb_map: Asset = self.assets_in["cmb_map"]
        self.beam_cfg = cfg.model.analysis.beam_pyilc
