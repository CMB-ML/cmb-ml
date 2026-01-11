import logging
import time 
import shutil

from omegaconf import DictConfig
from tqdm import tqdm

# import numpy as np
import healpy as hp
import pysm3.units as u

from cmbml.utils.planck_instrument import make_instrument, Instrument
from cmbml.core import BaseStageExecutor, Split, Asset

from cmbml.core.asset_handlers.qtable_handler import QTable # Import to register handler
from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap # Import for VS Code hints
from cmbml.utils.fits_inspection import get_field_types_from_fits


logger = logging.getLogger(__name__)


class NoiseFilterSimCreatorExecutor(BaseStageExecutor):
    """
    Nothing of interest happens here; refer to the particular noise creator (defined in the configs)!

    SimCreatorExecutor simply adds observations and noise.

    Attributes:
        out_obs_maps (Asset): The output asset for the observation maps.
        in_noise (Asset): The input asset for the noise map.
        in_sky (Asset): The input asset for the observation map (without noise).
        in_det_table (Asset): The input asset for the detector table.
        instrument (Instrument): The instrument configuration used for the simulation.

    Methods:
        execute() -> None:
            Overarching for all splits.
        process_split(split: Split) -> None:
            Overarching for all sims in a split.
        process_sim(split: Split, sim_num: int) -> None:
            Processes the given split and simulation number.
    """
    def __init__(self, cfg: DictConfig) -> None:
        # The following stage_str must match the pipeline yaml
        super().__init__(cfg, stage_str='make_sims_filter_noise')

        self.out_obs : Asset = self.assets_out['obs_maps']
        self.out_cmb : Asset = self.assets_out['cmb_map']
        out_obs_maps_handler: HealpyMap

        self.in_noise: Asset = self.assets_in['noise_maps']
        self.in_sky  : Asset = self.assets_in['sky_no_noise_maps']
        self.in_cmb  : Asset = self.assets_in['cmb_map']
        in_planck_bp_table: Asset = self.assets_in['planck_deltabandpass']
        in_maps_handler: HealpyMap
        in_ps_handler: QTable

        self.output_units = u.Unit(cfg.scenario.units)

        self.instrument: Instrument = make_instrument(cfg=cfg)
        self.include_cmb = cfg.model.sim.get("include_cmb", True)

        planck_det_info = in_planck_bp_table.read()
        self.planck_instrument: Instrument = make_instrument(cfg=cfg, 
                                                             det_info_override=planck_det_info)
        self.nside = cfg.scenario.nside
        lmax_ratio = cfg.model.sim.noise.lmax_ratio_planck_noise
        self.lmax = int(lmax_ratio * self.nside)
        self.beam_filters = {}

    def execute(self) -> None:
        """
        Adds noise and observations for all simulations.
        Hollow boilerplate.
        """
        logger.debug(f"Running {self.__class__.__name__} execute() method")
        self.setup_filters()
        self.default_execute()

    def setup_filters(self):
        for freq, det in self.instrument.dets.items():
            cmb_ml_fwhm = det.fwhm
            planck_fwhm = self.planck_instrument.dets[freq].fwhm
            cmb_ml_beam = hp.gauss_beam(cmb_ml_fwhm.to(u.rad).value,
                                        lmax=self.lmax)
            planck_beam = hp.gauss_beam(planck_fwhm.to(u.rad).value,
                                        lmax=self.lmax)
            some_filter = cmb_ml_beam / planck_beam
            self.beam_filters[freq] = some_filter

    def process_split(self, split: Split) -> None:
        """
        Adds noise and observations for all sims for a split.
        Hollow boilerplate.

        Args:
            split (Split): The split to process.
        """
        # logger.debug (f"Current time is {time.time()}")
        with tqdm(total=split.n_sims, desc=f"{split.name}: ", leave=False) as pbar:
            for sim in split.iter_sims():
                pbar.set_description(f"{split.name}: {sim:04d}")
                with self.name_tracker.set_context("sim_num", sim):
                    self.process_sim(split, sim_num=sim)
                pbar.update(1)

    def process_sim(self, split: Split, sim_num: int) -> None:
        """
        Adds noise and observations for a single simulation.

        Args:
            split (Split): The split to process. Needed for some configuration information.
            sim_num (int): The simulation number.
        """
        sim_name = self.name_tracker.sim_name()
        logger.debug(f"Creating simulation {split.name}:{sim_name}")
        for freq, detector in self.instrument.dets.items():
            with self.name_tracker.set_context("freq", freq):
                noise_maps = self.in_noise.read(map_field_strs=detector.fields)

                noise_maps = self.apply_filter(noise_maps, freq)

                sky_no_noise_maps = self.in_sky.read(map_field_strs=detector.fields)
                column_names = get_field_types_from_fits(self.in_noise.path)  # path requires being in freq context

            # Perform addition in-place 
            obs_maps = noise_maps.to(self.output_units, equivalencies=u.cmb_equivalencies(detector.cen_freq))
            obs_maps += sky_no_noise_maps.to(self.output_units, equivalencies=u.cmb_equivalencies(detector.cen_freq))

            with self.name_tracker.set_contexts(dict(freq=freq)):
                self.out_obs.write(data=obs_maps, column_names=column_names)
            logger.debug(f"For {split.name}:{sim_name}, {freq} GHz: done with channel")

        # Copy CMB map from input asset path to output asset path
        if self.include_cmb:
            cmb_in_path  = self.in_cmb.path
            cmb_out_path = self.out_cmb.path
            shutil.copy(cmb_in_path, cmb_out_path)

    def apply_filter(self, noise_maps, freq):
        unit = noise_maps.unit
        alms = hp.map2alm(noise_maps)  # TOOD: Use lmax here
        fl = self.beam_filters[freq]
        hp.almxfl(alms, fl, inplace=True)
        out_map = hp.alm2map(alms, self.nside)
        out_map = out_map[None, :]
        return u.Quantity(out_map, unit)
