from typing import Dict
import logging
import time 
import shutil

from omegaconf import DictConfig
from tqdm import tqdm
import numpy as np
import pysm3.units as u

from cmbml.utils.planck_instrument import make_instrument, Instrument
from cmbml.core import BaseStageExecutor, Split, Asset

from cmbml.core.asset_handlers import Config
from cmbml.core.asset_handlers.asset_handlers_base import PlainText
from cmbml.core.asset_handlers.qtable_handler import QTableHandler # Import to register handler
from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap # Import for VS Code hints
from cmbml.sims.random_seed_manager import SeedFactory
from cmbml.utils.patch_healpix import get_valid_ids


logger = logging.getLogger(__name__)


class ChoosePatchesExecutor(BaseStageExecutor):
    """
    SimCreatorExecutor simply adds observations and noise.

    Attributes:
        out_patch_id (Asset [Config]): The output asset for the observation maps.
        in_mask (Asset [HealpyMap]): The input asset for the mask map.
        in_det_table (Asset [QTable]): The input asset for the detector table.
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
        super().__init__(cfg, stage_str='choose_patches')

        self.out_all_patch_ids : Asset = self.assets_out['all_ids']
        self.out_patch_id : Asset = self.assets_out['patch_id']
        out_all_patch_ids_handler: Config
        out_patch_id_handler: PlainText

        self.in_mask: Asset = self.assets_in['mask']
        in_mask_handler: HealpyMap

        in_det_table: Asset  = self.assets_in['deltabandpass']
        in_det_table_handler: QTableHandler

        det_info = in_det_table.read()
        self.instrument: Instrument = make_instrument(cfg=cfg, det_info=det_info)

        self.patch_seed_factory = SeedFactory(cfg, cfg.model.patches.seed_template)
        self.nside_obs = cfg.scenario.nside
        self.nside_patch = cfg.model.patches.nside_patch
        self.mask_threshold = cfg.model.patches.mask_threshold

        # Placeholders
        self.valid_ids = None

    def get_valid_ids(self) -> None:
        """
        Gets the valid IDs based on the mask.
        """
        valid_ids = get_valid_ids(mask=self.in_mask.read(), 
                                  nside_obs=self.nside_obs, 
                                  nside_patches=self.nside_patch,
                                  threshold=self.mask_threshold)
        return valid_ids

    def execute(self) -> None:
        """
        Gets valid patch ID's, then runs for all splits.
        """
        logger.debug(f"Running {self.__class__.__name__} execute() method")
        self.valid_ids = self.get_valid_ids()
        self.default_execute()  # Sets name_tracker, calls process splits for all splits

    def process_split(self, split: Split) -> None:
        """
        Determines list of all patch IDs for a split, then processes each sim.

        Args:
            split (Split): The split to process.
        """
        n_ids = split.n_sims
        seed = self.patch_seed_factory.get_seed(split=split.name)

        rng = np.random.default_rng(seed)
        patch_ids = rng.choice(self.valid_ids, size=n_ids, replace=True)

        for i, sim in enumerate(split.iter_sims()):
            with self.name_tracker.set_context("sim_num", sim):
                self.process_sim(patch_id=patch_ids[i])
        all_patch_data = {
            'all_ids': list(self.valid_ids),
            'patch_ids': {n: patch_ids[n] for n in range(n_ids)}
        }
        self.out_all_patch_ids.write(data=all_patch_data)

    def process_sim(self, patch_id: int) -> None:
        """
        Writes patch ID to a config.

        Args:
            split (Split): The split to process. Needed for some configuration information.
            sim_num (int): The simulation number.
        """
        self.out_patch_id.write(data=patch_id)
