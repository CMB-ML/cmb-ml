import logging

import numpy as np
from tqdm import tqdm
import healpy as hp

from omegaconf import DictConfig

from cmbml.core import (
    BaseStageExecutor, 
    Split,
    Asset
    )
from cmbml.core.asset_handlers.ps_handler import NumpyPowerSpectrum
from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap # Import for typing hint
from cmbml.utils.physics_ps import get_auto_ps_result
from cmbml.utils.physics_beam import NoBeam, GaussianBeam
from cmbml.utils.physics_mask import downgrade_mask


logger = logging.getLogger(__name__)


class MakeRealPowerSpectrumExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="make_real_ps")

        self.out_auto_real: Asset = self.assets_out.get("auto_real", None)
        out_ps_handler: NumpyPowerSpectrum

        self.in_cmb_map_real: Asset = self.assets_in["cmb_map_real"]
        self.in_mask: Asset = self.assets_in.get("mask", None)
        self.in_mask_sm: Asset = self.assets_in.get("mask_sm", None)
        in_cmb_map_handler: HealpyMap

        # Basic parameters
        self.nside_out = self.cfg.scenario.nside

        self.lmax = cfg.model.get("lmax", None)
        if self.lmax is None:
            self.lmax = int(cfg.model.analysis.lmax_ratio * self.nside_out)
        self.anafast_iters = cfg.model.analysis.get("ps_anafast_iters")

        self.cmb_beam = cfg.scenario.cmb_beam

        # Prepare to load mask (in execute())
        self.mask_threshold = self.cfg.model.analysis.mask_threshold
        self.mask = None

        self.skip_mask = self.cfg.get("skip_real_ps_mask", None)
        if self.skip_mask:
            logger.warning("Skipping mask when getting Realization PS!")

        self.use_sm_mask = self.cfg.model.analysis.ps_use_smooth_mask

        self.beam_real = None

        self.use_pixel_weights = False

        if self.cfg.map_fields != "I":
            raise NotImplementedError("Only intensity maps are currently supported.")

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute().")
        if self.skip_mask:
            logger.warning("Skipping mask when getting Realization PS!")
            self.mask = self.get_masks()
        if self.cmb_beam == "min":
            raise NotImplementedError("Need to think through this option.")
        elif self.cmb_beam == 0:
            self.beam_real = NoBeam(self.lmax)
        else:
            self.beam_real = GaussianBeam(self.cmb_beam, self.lmax)
        self.default_execute()

    def get_masks(self):
        mask = None
        with self.name_tracker.set_context("src_root", self.cfg.local_system.assets_dir):
            if self.use_sm_mask:
                logger.info(f"Using mask from {self.in_mask_sm.path}")
                mask = self.in_mask_sm.read(map_fields=self.in_mask_sm.use_fields)[0]
            else:
                logger.info(f"Using mask from {self.in_mask.path}")
                mask = self.in_mask.read(map_fields=self.in_mask.use_fields)[0]
        if hp.npix2nside(mask.size) != self.nside_out:
            mask = downgrade_mask(mask, self.nside_out, threshold=self.mask_threshold)
        return mask

    def process_split(self, 
                      split: Split) -> None:
        logger.info(f"Running {self.__class__.__name__} process_split() for split: {split.name}.")
        for sim in tqdm(split.iter_sims()):
            with self.name_tracker.set_context("sim_num", sim):
                self.process_sim()

    def process_sim(self) -> None:
        # Get power spectrum for realization
        real_map: np.ndarray = self.in_cmb_map_real.read()
        if real_map.shape[0] == 3 and self.map_fields == "I":
            real_map = real_map[0]
        self.make_real_ps(real_map)

    def make_real_ps(self, real_map):
        auto_real_ps = get_auto_ps_result(real_map,
                                          mask=self.mask,  # If mask loading was skipped, this is None.
                                          lmax=self.lmax,
                                          n_iter=self.anafast_iters,
                                          beam=self.beam_real,
                                          is_convolved=True)
        ps = auto_real_ps.deconv_dl
        self.out_auto_real.write(data=ps.value)
