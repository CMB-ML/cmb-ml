from typing import Dict, Union
from pathlib import Path
import logging

import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np
import pysm3
import pysm3.units as u
from tqdm import tqdm

from cmbml.sims.random_seed_manager import SeedFactory
from cmbml.utils.planck_instrument import make_instrument, Instrument

from cmbml.core import (
    BaseStageExecutor,
    Split,
    Asset, AssetWithPathAlts
)

from cmbml.core.asset_handlers.ps_handler import CambPowerSpectrum, NumpyPowerSpectrum # Import for typing hint
from cmbml.utils.physics_ps import dl_to_cl, cl_to_dl
import healpy as hp


logger = logging.getLogger(__name__)


class CheapRealizationPSExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig, stage_str='make_cheap_real_ps') -> None:
        # The following stage_str must match the pipeline yaml
        super().__init__(cfg, stage_str=stage_str)

        self.out_real_ps: Asset = self.assets_out['cmb_ps']
        out_cmb_ps_handler: NumpyPowerSpectrum

        self.in_cmb_ps: AssetWithPathAlts = self.assets_in['cmb_ps']
        in_cmb_ps_handler: CambPowerSpectrum

        # Initialize constants from configs
        self.nside_out = cfg.scenario.nside
        logger.info(f"CMB maps will be generated at nside = {self.nside_out}")
        self.output_units = u.Unit(cfg.scenario.units)
        logger.info(f"Output units are {self.output_units**2}")

        self.instrument: Instrument = make_instrument(cfg=cfg)

        self.cmb_seed_factory = SeedFactory(cfg.model.sim.cmb.seed_template_map)

        self.output_lmax = cfg.model.sim.downgrade_lmax

    def execute(self) -> None:
        """
        Creates simulations for all sims within all splits.

        Sets up the Sky object just once
           - Make placeholder object for CMB (others could be added here)
           - Preset strings are passed here
        Thus, components from preset strings are created here, once, for all simulations
        """
        logger.debug(f"Running {self.__class__.__name__} execute() method")
        self.default_execute()

    def process_split(self, split: Split) -> None:
        """
        Processes all sims for a split, making simulations.
        Hollow boilerplate.

        Args:
            split (Split): The split to process.
        """
        with tqdm(total=split.n_sims, desc=f"{split.name}: ", leave=False) as pbar:
            for sim in split.iter_sims():
                pbar.set_description(f"{split.name}: {sim:04d}")
                with self.name_tracker.set_context("sim_num", sim):
                    self.process_sim(split, sim_num=sim)
                pbar.update(1)

    def process_sim(self, split: Split, sim_num: int) -> None:
        """
        Produces a single simulation. Expects the sky object to be initialized.

        Args:
            split (Split): The split to process. Needed for some configuration information.
            sim_num (int): The simulation number.
        """
        sim_name = self.name_tracker.sim_name()  # For logging and seed generation
        logger.debug(f"Creating simulation {split.name}:{sim_name}")

        # ps_path = self.in_cmb_ps.path_alt if split.ps_fidu_fixed else self.in_cmb_ps.path
        thry_dl = self.in_cmb_ps.read(use_alt_path=split.ps_fidu_fixed)
        ells = np.arange(self.output_lmax+1)
        thry_cl = dl_to_cl(thry_dl, ells=ells)

        cmb_seed = self.cmb_seed_factory.get_seed(split=split,
                                                    sim=sim_name)
        np.random.seed(cmb_seed)
        real_map = hp.synfast(cls=thry_cl, nside=self.nside_out)
        real_cl = hp.anafast(real_map, lmax=self.output_lmax)
        real_dl = cl_to_dl(real_cl, ells=ells)

        self.out_real_ps.write(data=real_dl)
        logger.debug(f"For {split.name}:{sim_name}, done with simulation")
