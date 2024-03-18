from typing import Dict, List
import logging

import numpy as np

from omegaconf import DictConfig

from ...core import (
    BaseStageExecutor,
    Split,
    ExperimentParameters,
    Asset
    )


logger = logging.getLogger(__name__)


class PreprocessMakeExtremaExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig, experiment: ExperimentParameters) -> None:
        logger.debug("Initializing Petroff PreprocessMakeExtremaExecutor")
        # The following string must match the pipeline yaml
        self.stage_str = "make_normalization"
        super().__init__(cfg, experiment)

        self.channels: List[str] = ["cmb", *self.experiment.detector_freqs]

        self.out_norm_file: Asset = self.assets_out["norm_file"]
        self.in_cmb_map: Asset = self.assets_in["cmb_map"]
        self.in_obs_maps: Asset = self.assets_in["obs_maps"]

    def execute(self) -> None:
        logger.debug("PreprocessMakeExtremaExecutor execute() method.")
        # Defining extrema at the scope of the stage: we want extrema of all maps across splits
        #    Note that some channels won't use all fields (e.g. 545, 857 only have intensity)
        extrema = {detector: {} for detector in self.channels}
        for split in self.splits:
            with self.name_tracker.set_context("split", split.name):
                self.search_split_contents(split, extrema)
        pass
        # raise Exception("Jim: you are here.")
        self.out_norm_file.write(extrema)

    def search_split_contents(self, 
                              split: Split, 
                              extrema: Dict[str, Dict]) -> None:
        # A split is composed of simulations
        for sim in split.iter_sims():
            with self.name_tracker.set_context("sim_num", sim):
                self.search_sim_contents(extrema)

    def search_sim_contents(self, 
                            extrema: Dict[str, Dict]) -> None:
        # A simulation has many maps: observations (made by many detectors) and cmb
        in_obs_maps_data = self.in_obs_maps.read()
        for detector in self.channels:
            if detector == "cmb":
                map_data = self.in_cmb_map.read()
            else:
                map_data = in_obs_maps_data[detector]
            self.search_map_file_contents(extrema, map_data, detector)

    def search_map_file_contents(self, 
                                 extrema: Dict[str, Dict], 
                                 map_data: np.ndarray, 
                                 channel: str) -> None:
        for field_n in range(map_data.shape[0]):
            field_data = map_data[field_n]
            field_char = self.experiment.map_fields[field_n]
            self.update_extrema_asymmetric(extrema, field_data, channel, field_char)

    def update_extrema_asymmetric(self,
                                  extrema: Dict[str, Dict],
                                  map_data: np.ndarray,
                                  channel: str,
                                  field_char: str) -> None:
        # "Asymmetric" in that there's no assumption of symmetry. Cf Petroff's method.
        if field_char not in extrema[channel].keys():
            extrema[channel][field_char] = {}

        min_val = min(map_data)
        max_val = max(map_data)
        old_min = extrema[channel][field_char].get("min_val", None)
        if old_min is None or min_val < old_min:
            extrema[channel][field_char]["min_val"] = min_val
        old_max = extrema[channel][field_char].get("max_val", None)
        if old_max is None or max_val > old_max:
            extrema[channel][field_char]["max_val"] = max_val

#     def update_extrema_symmetric(extrema, map_data, channel, field) -> None:
#         # "Symmetric" in that there's the following assumptions:
#         #    max values are positive, 
#         #    min values are negative
#         #    the absolute value of the max is roughly the absolute value of the min
#         raise NotImplementedError("Symmetric normalization not yet implemented.")
#         # TODO: Look up how Petroff did this. I can't remember if he assumed that min value was negative.
