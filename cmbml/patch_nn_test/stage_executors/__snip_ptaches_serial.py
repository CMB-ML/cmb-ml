from typing import Dict, List, Any
import logging
from functools import partial

import numpy as np

from omegaconf import DictConfig

from tqdm import tqdm

from cmbml.core import BaseStageExecutor, Split, Asset
from cmbml.utils import make_instrument, Instrument
from cmbml.core.asset_handlers import Config    # Import for typing hint
from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap  # Import for typing hint
from cmbml.core.asset_handlers.handler_npymap import NumpyMap
from cmbml.utils.map_fields_helper import map_field_str2int


logger = logging.getLogger(__name__)


class SerialSnipPatchesExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="prep_patches")

        self.instrument: Instrument = make_instrument(cfg=cfg)

        self.out_cmb_patch: Asset = self.assets_out["cmb_patch"]
        self.out_obs_patch: Asset = self.assets_out["obs_patch"]
        out_patches_handler: NumpyMap

        self.in_cmb_map: Asset = self.assets_in["cmb_map"]
        self.in_obs_maps: Asset = self.assets_in["obs_maps"]
        self.in_norm_file: Asset = self.assets_in["norm_file"]
        self.in_lut: Asset = self.assets_in["lut"]
        self.in_patch_id: Asset = self.assets_in["patch_id"]
        in_cmb_map_handler: HealpyMap
        in_obs_map_handler: HealpyMap
        in_norm_file_handler: Config
        in_lut_handler: NumpyMap
        in_patch_id_handler: Config

        self.lut = None
        self.extrema = None

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        self.ensure_splits()
        self.get_lut()
        self.get_extrema()
        for split in self.splits:
            logger.info(f"{self.__class__.__name__} preprocessing {split.name}.")
            with self.name_tracker.set_context("split", split.name):
                self.process_split(split)

    def get_lut(self) -> None:
        self.lut = self.in_lut.read()

    def get_extrema(self) -> None:
        self.extrema = self.in_norm_file.read()

    def process_split(self, 
                      split: Split) -> None:
        # A split is composed of simulations
        for sim in tqdm(split.iter_sims()):
            logger.debug(f"Preprocessing sim {sim}.")
            with self.name_tracker.set_context("sim_num", sim):
                self.process_sim()

    def process_sim(self) -> None:
        patch_id = int(self.in_patch_id.read())
        r_ids = self.lut[patch_id]

        cmb_map = self.in_cmb_map.read()
        extrema = self.extrema["cmb"]
        self.save_patch_from_map(
                                 cmb_map, 
                                 r_ids,
                                 extrema,
                                 self.out_cmb_patch
                                 )

        for freq in self.instrument.dets.keys():
            with self.name_tracker.set_context("freq", freq):
                obs_map = self.in_obs_maps.read()
                self.save_patch_from_map(obs_map, r_ids, self.extrema[freq], self.out_obs_patch)

    def save_patch_from_map(self,
                            map_data: np.ndarray,  # Or Astropy Quantity
                            r_ids: List[int],
                            extrema: Dict[str, Dict[str, Any]],
                            out_asset: Asset,
                            ) -> None:
        """
        From map data, extract patches, scale, and save them to an asset.
        """
        map_patches = []
        for field_str in self.map_fields:
            field_int = map_field_str2int(field_str)
            patch = map_data[field_int][r_ids]
            patch = minmax_scale(patch, 
                                 extrema[field_str]["vmin"], 
                                 extrema[field_str]["vmax"])
            map_patches.append(patch.value)
        map_patches = np.array(map_patches)
        out_asset.write(data=map_patches)


def minmax_scale(data, vmin, vmax):
    return (data - vmin) / (vmax - vmin)
