from typing import Dict, List
import logging

import numpy as np

from omegaconf import DictConfig

from tqdm import tqdm

from cmbml.core import BaseStageExecutor, Split, Asset
from cmbml.utils import make_instrument, Instrument
from cmbml.core.asset_handlers import Config    # Import for typing hint
from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap  # Import for typing hint


logger = logging.getLogger(__name__)


class SerialPreprocessMakeExtremaExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="make_normalization")

        instrument: Instrument = make_instrument(cfg=cfg)
        self.channels = ["cmb", *instrument.dets.keys()]

        self.out_norm_file: Asset = self.assets_out["norm_file"]
        out_norm_file_handler: Config

        self.in_cmb_map: Asset = self.assets_in["cmb_map"]
        self.in_obs_maps: Asset = self.assets_in["obs_maps"]
        in_cmb_map_handler: HealpyMap
        in_obs_map_handler: HealpyMap

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        self.ensure_splits()
        # Defining extrema at the scope of the stage: we want extrema of all maps across splits
        #    Note that some channels won't use all fields (e.g. 545, 857 only have intensity)
        extrema = {detector: {} for detector in self.channels}
        for split in self.splits:
            logger.info(f"{self.__class__.__name__} scanning split {split.name}.")
            with self.name_tracker.set_context("split", split.name):
                self.search_split_contents(split, extrema)
        self.out_norm_file.write(data=extrema)

    def search_split_contents(self, 
                              split: Split, 
                              extrema: Dict[str, Dict]) -> None:
        # A split is composed of simulations
        for sim in tqdm(split.iter_sims()):
            logger.info(f"Scanning sim {sim}.")
            with self.name_tracker.set_context("sim_num", sim):
                self.search_sim_contents(extrema)

    def search_sim_contents(self, 
                            extrema: Dict[str, Dict]) -> None:
        # A simulation has many maps: observations (made at many frequencies) and cmb
        for freq in self.channels:
            if freq == "cmb":
                map_data = self.in_cmb_map.read()
            else:
                with self.name_tracker.set_context("freq", freq):
                    in_obs_maps_data = self.in_obs_maps.read()
                map_data = in_obs_maps_data
            self.search_map_file_contents(extrema, map_data, freq)

    def search_map_file_contents(self, 
                                 extrema: Dict[str, Dict], 
                                 map_data: np.ndarray, 
                                 channel: str) -> None:
        for field_n in range(map_data.shape[0]):
            try:
                field_data = map_data[field_n]
                field_char = self.cfg.scenario.map_fields[field_n]
                self.update_extrema(extrema, field_data, channel, field_char)
            except IndexError:
                continue

    def update_extrema(self,
                       extrema: Dict[str, Dict],
                       map_data: np.ndarray,
                       channel: str,
                       field_char: str) -> None:
        if field_char not in extrema[channel].keys():
            extrema[channel][field_char] = {}

        min_val = map_data.min()
        max_val = map_data.max()
        old_min = extrema[channel][field_char].get("min_val", None)
        if old_min is None or min_val < old_min:
            extrema[channel][field_char]["min_val"] = min_val
        old_max = extrema[channel][field_char].get("max_val", None)
        if old_max is None or max_val > old_max:
            extrema[channel][field_char]["max_val"] = max_val
