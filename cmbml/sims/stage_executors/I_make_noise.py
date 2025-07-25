from typing import Dict, Union
import logging
import time 

from omegaconf import DictConfig

from tqdm import tqdm

from cmbml.core import BaseStageExecutor, Split, Asset
from cmbml.core.asset_handlers.qtable_handler import QTableHandler # Import to register handler
from cmbml.core.asset_handlers.psmaker_handler import NumpyPowerSpectrum # Import for typing hint
from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap # Import for VS Code hints

from cmbml.sims.random_seed_manager import SeedFactory
from cmbml.utils.planck_instrument import make_instrument, Instrument
from cmbml.sims.physics_instrument import get_noise_class


logger = logging.getLogger(__name__)


class NoiseMapCreatorExecutor(BaseStageExecutor):
    """
    Nothing of interest happens here; refer to the particular noise creator (defined in the configs)!

    SimCreatorExecutor is responsible for generating the simulated maps for a given simulation scenario.

    Attributes:
        out_noise_maps (Asset): The output asset for the noise maps.
        in_noise_cache (Asset): The input asset for the noise cache.
        in_cmb_ps (AssetWithPathAlts): The input asset for the CMB power spectra.
        in_det_table (Asset): The input asset for the detector table.
        instrument (Instrument): The instrument configuration used for the simulation.
        cmb_seed_factory (SimLevelSeedFactory): The seed factory for the CMB.
        noise_seed_factory (FieldLevelSeedFactory): The seed factory for the noise.
        nside_sky (int): The nside for the sky.
        nside_out (int): The nside for the output maps.
        lmax_out (int): The lmax for the output maps.
        units (str): The units for the output maps.
        preset_strings (List[str]): The preset strings for the sky.
        output_units (str): The output units for the sky.
        cmb_factory (CMBFactory): The factory for the CMB.
        sky (pysm3.Sky): The sky object for the simulation.

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
        super().__init__(cfg, stage_str='make_noise')

        self.out_noise_maps: Asset = self.assets_out['noise_maps']
        out_noise_maps_handler: HealpyMap

        self.in_noise_cache: Asset = self.assets_in['scale_cache']
        in_det_table: Asset = self.assets_in['deltabandpass']
        in_noise_cache_handler: Union[HealpyMap, NumpyPowerSpectrum]
        in_det_table_handler: QTableHandler

        self.nside_out = cfg.scenario.nside
        logger.info(f"Noise will be created at nside_out = {self.nside_out}")
        self.units = cfg.scenario.units
        logger.info(f"Noise will have units of {self.units}")
        self.output_units = cfg.scenario.units

        det_info = in_det_table.read()
        self.instrument: Instrument = make_instrument(cfg=cfg, det_info=det_info)

        self.noise_seed_factory   = SeedFactory(cfg, cfg.model.sim.noise.seed_template)
        NoiseMaker                = get_noise_class(cfg.model.sim.noise.noise_type)
        self.noise_maker          = NoiseMaker(cfg, self.name_tracker, self.in_noise_cache)

    def execute(self) -> None:
        """
        Creates noise for all sims within all splits.
        Hollow boilerplate.
        """
        logger.debug(f"Running {self.__class__.__name__} execute() method")
        self.default_execute()

    def process_split(self, split: Split) -> None:
        """
        Processes all sims for a split, making noise simulations.
        Hollow boilerplate.

        Args:
            split (Split): The split to process.
        """
        logger.debug(f"Current time is{time.time()}")
        with tqdm(total=split.n_sims, desc=f"{split.name}: ", leave=False) as pbar:
            for sim in split.iter_sims():
                pbar.set_description(f"{split.name}: {sim:04d}")
                with self.name_tracker.set_context("sim_num", sim):
                    self.process_sim(split, sim_num=sim)
                pbar.update(1)

    def process_sim(self, split: Split, sim_num: int) -> None:
        """
        Produces a noise for a single simulation. Calls the noise maker to create the noise map.

        Args:
            split (Split): The split to process. Needed for some configuration information.
            sim_num (int): The simulation number.
        """
        sim_name = self.name_tracker.sim_name()  # For logging and seed generation
        logger.debug(f"Creating simulation {split.name}:{sim_name}")
        for freq, detector in self.instrument.dets.items():
            noise_seed   = self.noise_seed_factory.get_seed(split=split.name, 
                                                            sim=f"{sim_num:04}",  # Needed to duplicate seed integers
                                                            freq=freq)
            noise_map    = self.noise_maker.get_noise_map(detector, noise_seed)
            column_names = [f"{stokes}_STOKES" for stokes in detector.fields]

            with self.name_tracker.set_contexts(dict(freq=freq)):
                self.out_noise_maps.write(data=noise_map, column_names=column_names)
            logger.debug(f"For {split.name}:{sim_name}, {freq} GHz: done with channel")
