import logging
import time 
import shutil

from omegaconf import DictConfig
from tqdm import tqdm

import pysm3.units as u

from cmbml.utils.planck_instrument import make_instrument, Instrument
from cmbml.core import BaseStageExecutor, Split, Asset

from cmbml.core.asset_handlers.qtable_handler import QTableHandler # Import to register handler
from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap # Import for VS Code hints
from cmbml.utils.fits_inspection import get_field_types_from_fits


logger = logging.getLogger(__name__)


class SimCreatorExecutor(BaseStageExecutor):
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
        super().__init__(cfg, stage_str='make_sims')

        self.out_obs : Asset = self.assets_out['obs_maps']
        self.out_cmb : Asset = self.assets_out['cmb_map']
        out_obs_maps_handler: HealpyMap

        self.in_noise: Asset = self.assets_in['noise_maps']
        self.in_sky  : Asset = self.assets_in['sky_no_noise_maps']
        self.in_cmb  : Asset = self.assets_in['cmb_map']
        in_maps_handler: HealpyMap

        in_det_table: Asset  = self.assets_in['deltabandpass']
        in_det_table_handler: QTableHandler

        self.output_units = u.Unit(cfg.scenario.units)

        det_info = in_det_table.read()
        self.instrument: Instrument = make_instrument(cfg=cfg, det_info=det_info)

    def execute(self) -> None:
        """
        Adds noise and observations for all simulations.
        Hollow boilerplate.
        """
        logger.debug(f"Running {self.__class__.__name__} execute() method")
        self.default_execute()

    def process_split(self, split: Split) -> None:
        """
        Adds noise and observations for all sims for a split.
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
                sky_no_noise_maps = self.in_sky.read(map_field_strs=detector.fields)
                column_names = get_field_types_from_fits(self.in_noise.path)  # path requires being in freq context

            # Perform addition in-place 
            obs_maps = 1e-6 * noise_maps.to(self.output_units, equivalencies=u.cmb_equivalencies(detector.cen_freq))
            obs_maps += sky_no_noise_maps.to(self.output_units, equivalencies=u.cmb_equivalencies(detector.cen_freq))

            with self.name_tracker.set_contexts(dict(freq=freq)):
                self.out_obs.write(data=obs_maps, column_names=column_names)
            logger.debug(f"For {split.name}:{sim_name}, {freq} GHz: done with channel")

        # Copy CMB map from input asset path to output asset path
        cmb_in_path  = self.in_cmb.path
        cmb_out_path = self.out_cmb.path
        shutil.copy(cmb_in_path, cmb_out_path)
