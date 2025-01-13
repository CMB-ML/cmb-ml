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


class SnipConfigExecutor(BaseStageExecutor):
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
        super().__init__(cfg, stage_str='patches_cfg')

        self.out_all_patch_ids : Asset = self.assets_out['all_ids']
        self.out_patch_id : Asset = self.assets_out['patch_id']
        out_all_patch_ids_handler: Config
        out_patch_id_handler: PlainText

        self.in_mask: Asset = self.assets_in['mask']
        in_mask_handler: HealpyMap

        in_det_table: Asset  = self.assets_in['planck_deltabandpass']
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


class SnipExecutor(BaseStageExecutor):
    """
    SnipExecutor cuts out a patch from the models for training with pre-snipped maps.
    
    If your model needs maps pre-processed before snipping, this should be done in an
    executor for your particular model. I've tried to create function calls for the snipping
    in <other module>.

    Attributes: 
        TODO: UPDATE THIS
        out_obs_maps (Asset): The output asset for the observation maps.
        in_noise (Asset): The input asset for the noise map.
        in_sky (Asset): The input asset for the observation map (without noise).
        in_det_table (Asset): The input asset for the detector table.
        instrument (Instrument): The instrument configuration used for the simulation.

    Methods:
        TODO: UPDATE THIS?
        execute() -> None:
            Overarching for all splits.
        process_split(split: Split) -> None:
            Overarching for all sims in a split.
        process_sim(split: Split, sim_num: int) -> None:
            Processes the given split and simulation number.
    """
    def __init__(self, cfg: DictConfig) -> None:
        # The following stage_str must match the pipeline yaml
        super().__init__(cfg, stage_str='snip_patches')
        raise NotImplementedError("This executor is not yet implemented.")

    #     self.out_patch_id : Asset = self.assets_out['patch_id']
    #     self.out_obs : Asset = self.assets_out['obs_maps']
    #     self.out_cmb : Asset = self.assets_out['cmb_map']
    #     out_obs_maps_handler: HealpyMap

    #     self.in_obs  : Asset = self.assets_in['obs_maps']
    #     self.in_cmb  : Asset = self.assets_in['cmb_map']
    #     in_maps_handler: HealpyMap

    #     in_det_table: Asset  = self.assets_in['planck_deltabandpass']
    #     in_det_table_handler: QTableHandler

    #     det_info = in_det_table.read()
    #     self.instrument: Instrument = make_instrument(cfg=cfg, det_info=det_info)

    #     self.patch_seed_factory = SeedFactory(cfg, 'patches')

    #     do_all_patches = cfg.model.patches.do_all_patches
    #     self.do_all_patches = self.get_do_all_patches(do_all_patches)

    # def get_do_all_patches(self, do_all_patches: Dict[str, bool]) -> Dict[str, bool]:
    #     """
    #     Gets the do_all_patches dictionary from the configuration.
    #         do_all_patches is a dictionary that specifies whether to do all patches for a given split.
    #         The intention was to have the Training set get single patches, 
    #         and the Test and Validation sets get all patches.

    #         It may be better to only get patches for the Train set, then provide
    #         a different dataloader for the Test and Validation sets.

    #     Args:
    #         do_all_patches (Dict[str, bool]): The dictionary from the configuration.

    #     Returns:
    #         Dict[str, bool]: The dictionary with the default values added.
    #     """
    #     out = {}
    #     for key in do_all_patches.keys():
    #         if key in self.splits.keys():
    #             out[key] = do_all_patches[key]
    #         elif key.capitalize in self.splits.keys():
    #             out[key.capitalize()] = do_all_patches[key]
    #         else:
    #             # TODO: update to handle Test# type splits (when implemented)
    #             raise ValueError(f"Key {key} not found in splits")
    #     return out

    # def execute(self) -> None:
    #     """
    #     Adds noise and observations for all simulations.
    #     Hollow boilerplate.
    #     """
    #     logger.debug(f"Running {self.__class__.__name__} execute() method")
    #     self.default_execute()  # Sets name_tracker, calls process splits for all splits

    # def process_split(self, split: Split) -> None:
    #     """
    #     Adds noise and observations for all sims for a split.
    #     Hollow boilerplate.

    #     Args:
    #         split (Split): The split to process.
    #     """
    #     logger.debug(f"Current time is{time.time()}")
    #     with tqdm(total=split.n_sims, desc=f"{split.name}: ", leave=False) as pbar:
    #         for sim in split.iter_sims():
    #             pbar.set_description(f"{split.name}: {sim:04d}")
    #             with self.name_tracker.set_context("sim_num", sim):
    #                 self.process_sim(split, sim_num=sim)
    #             pbar.update(1)

    # def process_sim(self, split: Split, sim_num: int) -> None:
    #     """
    #     Adds noise and observations for a single simulation.

    #     Args:
    #         split (Split): The split to process. Needed for some configuration information.
    #         sim_num (int): The simulation number.
    #     """
    #     pass
    #     # sim_name = self.name_tracker.sim_name()
    #     # logger.debug(f"Creating simulation {split.name}:{sim_name}")
    #     # for freq, detector in self.instrument.dets.items():
    #     #     with self.name_tracker.set_context("freq", freq):
    #     #         noise_maps = self.in_noise.read(map_field_strs=detector.fields)
    #     #         sky_no_noise_maps = self.in_sky.read(map_field_strs=detector.fields)
    #     #         column_names = get_field_types_from_fits(self.in_noise.path)  # path requires being in freq context

    #     #     # Perform addition in-place 
    #     #     obs_maps = noise_maps.to(self.output_units, equivalencies=u.cmb_equivalencies(detector.cen_freq))
    #     #     obs_maps += sky_no_noise_maps.to(self.output_units, equivalencies=u.cmb_equivalencies(detector.cen_freq))

    #     #     with self.name_tracker.set_contexts(dict(freq=freq)):
    #     #         self.out_obs.write(data=obs_maps, column_names=column_names)
    #     #     logger.debug(f"For {split.name}:{sim_name}, {freq} GHz: done with channel")

    #     # # Copy CMB map from input asset path to output asset path
    #     # cmb_in_path  = self.in_cmb.path
    #     # cmb_out_path = self.out_cmb.path
    #     # shutil.copy(cmb_in_path, cmb_out_path)
