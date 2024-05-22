from typing import List, Dict
import logging

import numpy as np

from omegaconf import DictConfig

from src.core import (
    BaseStageExecutor, 
    Split,
    Asset
    )


logger = logging.getLogger(__name__)


class SerialPreprocessExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        raise NotImplementedError("Serial Executor not implemented.")
        logger.debug("Initializing Petroff PreprocessExecutor")
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="preprocess")

        self.out_cmb_asset: Asset = self.assets_out["cmb_map"]
        self.out_obs_assets: Asset = self.assets_out["obs_maps"]

        self.in_norm_file: Asset = self.assets_in["norm_file"]
        self.in_cmb_asset: Asset = self.assets_in["cmb_map"]
        self.in_obs_assets: Asset = self.assets_in["obs_maps"]

    def execute(self) -> None:
        # Remove this function
        logger.debug(f"Running {self.__class__.__name__} execute() method.")
        super().execute()

    def process_split(self, 
                      split: Split) -> None:
        logger.info(f"Executing PreprocessExecutor process_split() for split: {split.name}.")
        
        logger.debug(f"Reading norm_file from: {self.in_norm_file.path}")
        extrema = self.in_norm_file.read()
        for sim in split.iter_sims():
            with self.name_tracker.set_context("sim_num", sim):
                self.process_sim(extrema)

    def process_sim(self, 
                    all_extrema) -> None:
        in_cmb_map = self.in_cmb_asset.read()
        normed_map = self.normalize_map_file(in_cmb_map, channel_extrema=all_extrema['cmb'])
        self.out_cmb_asset.write(normed_map)

        in_obs_assets: Dict[str, np.ndarray] = self.in_obs_assets.read()
        for detector, obs_map_file_contents in in_obs_assets.items():
            normed_map = self.normalize_map_file(obs_map_file_contents,
                                                 channel_extrema=all_extrema[detector])
            self.out_obs_assets.write({detector: normed_map})

    def normalize_map_file(self, 
                           map_data: np.ndarray, 
                           channel_extrema: Dict[str, Dict[str, float]]) -> List[np.ndarray]:
        normed_map = []
        for field_n in range(map_data.shape[0]):
            field_char = self.experiment.map_fields[field_n]
            field_data = map_data[field_n]
            field_extrema = channel_extrema[field_char]
            normed_map.append(self.normalize(field_data, field_extrema))

        return normed_map

    def normalize(self, in_map, extrema):
        norm_min = extrema['min_val']
        norm_max = extrema['max_val']

        out_map = (in_map - norm_min) / (norm_max - norm_min)
        return out_map
