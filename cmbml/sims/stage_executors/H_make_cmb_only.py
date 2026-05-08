import logging

import healpy as hp
import numpy as np
from astropy.units import Quantity
from omegaconf import DictConfig
import pysm3.units as u
from tqdm import tqdm

from cmbml.sims.random_seed_manager import SeedFactory

from cmbml.core import BaseStageExecutor, Split, Asset, AssetWithPathAlts
from cmbml.core.asset_handlers.ps_handler import DictCambPowerSpectrum
from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap
from cmbml.utils.physics_ps import dl_to_cl


logger = logging.getLogger(__name__)


class CMBCreatorExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig, stage_str='make_cmb_only') -> None:
        # The following stage_str must match the pipeline yaml
        super().__init__(cfg, stage_str=stage_str)

        self.out_cmb_map: Asset = self.assets_out['cmb_map']
        out_cmb_map_handler: HealpyMap

        self.in_cmb_ps: AssetWithPathAlts = self.assets_in['cmb_ps']
        in_cmb_ps_handler: DictCambPowerSpectrum

        self.nside_out = cfg.scenario.nside
        logger.info(f"Simulations will be output at nside_out = {self.nside_out}")
        self.lmax = int(cfg.cmb_lmax_fac * self.nside_out)
        logger.info(f"Simulation maps will be created with lmax = {self.lmax}")
        self.output_units = cfg.scenario.units
        logger.info(f"Output units are {self.output_units}")

        self.cmb_seed_factory = SeedFactory(cfg.model.sim.cmb.seed_template_map)

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute() method")
        self.default_execute()

    def process_split(self, split: Split) -> None:
        with tqdm(total=split.n_sims, desc=f"{split.name}: ", leave=False) as pbar:
            for sim in split.iter_sims():
                pbar.set_description(f"{split.name}: {sim:04d}")
                with self.name_tracker.set_context("sim_num", sim):
                    self.process_sim(split)
                pbar.update(1)

    def process_sim(self, split: Split) -> None:
        sim_name = self.name_tracker.sim_name()  # For logging and seed generation

        cmb_seed = self.cmb_seed_factory.get_seed(split=split,
                                                  sim=sim_name)

        ps = self.in_cmb_ps.read(fields=["L", "TT"], use_alt_path=split.ps_fidu_fixed)
        ells = ps["L"]
        dl_tt = ps["TT"]
        cl = dl_to_cl(dl_tt, ells)

        np.random.seed(cmb_seed)
        m = hp.synfast(cl, nside=self.nside_out, lmax=self.lmax)
        m = Quantity(m, u.uK_CMB)  # Fixed, per CAMB convention

        self.out_cmb_map.write(data=m)
