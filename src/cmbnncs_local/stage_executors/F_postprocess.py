from typing import List, Dict
import logging

import numpy as np

from omegaconf import DictConfig

from core import (
    BaseStageExecutor, 
    Split,
    ExperimentParameters,
    Asset
    )
from cmbnncs.spherical import piecePlanes2spheres
# from ..handler_npymap import NumpyMap  # Must import to register the AssetHandler
from tqdm import tqdm


logger = logging.getLogger(__name__)


class PostprocessExecutor(BaseStageExecutor):
    def __init__(self,
                 cfg: DictConfig,
                 experiment: ExperimentParameters) -> None:
        logger.debug("Initializing CMBNNCS PostprocessExecutor")
        # TODO: remove self.stage_str; pass it as a parameter to the super.init()
        # The following string must match the pipeline yaml
        self.stage_str = "postprocess"
        super().__init__(cfg, experiment)

        self.out_cmb_asset: Asset = self.assets_out["cmb_map_post"]

        self.in_norm_file: Asset = self.assets_in["norm_file"]
        self.in_cmb_asset: Asset = self.assets_in["cmb_map"]

        self.model_epochs = cfg.training.postprocess.epoch

    def execute(self) -> None:
        # Remove this function
        logger.debug(f"Executing PostprocessExecutor execute()")
        for epoch in self.model_epochs:
            # logger.debug(f"Executing PostprocessExecutor execute() for epoch: {epoch}")
            with self.name_tracker.set_context("epoch", epoch):
                super().execute()

    def process_split(self, 
                      split: Split) -> None:
        logger.info(f"Executing PostprocessExecutor process_split() for epoch: {self.name_tracker.context['epoch']}, split: {split.name}.")
        logger.debug(f"Reading norm_file from: {self.in_norm_file.path}")
        scale_factors = self.in_norm_file.read()
        for sim in tqdm(split.iter_sims()):
            with self.name_tracker.set_context("sim_num", sim):
                self.process_sim(scale_factors)

    def process_sim(self, scale_factors) -> None:
        in_cmb_map = self.in_cmb_asset.read()
        scaled_map = self.unnormalize_map_file(in_cmb_map, scale_factors=scale_factors['cmb'])
        self.out_cmb_asset.write(scaled_map)

    def unnormalize_map_file(self, 
                           map_data: np.ndarray, 
                           scale_factors: Dict[str, Dict[str, float]]) -> List[np.ndarray]:
        # processed_maps = []
        # for field_n in range(map_data.shape[0]):
        #     field_char = self.experiment.map_fields[field_n]
        #     field_scale = scale_factors[field_char]
        #     field_data = map_data[field_n]
        #     scaled_map = self.unnormalize(field_data, field_scale)
        #     demangled_map = piecePlanes2spheres(scaled_map)
        #     processed_maps.append(demangled_map)
        # return processed_maps
        field_scale = {'scale': 5}
        map_data = piecePlanes2spheres(map_data)
        map_data = self.unnormalize(map_data, field_scale)
        # demangled_map = piecePlanes2spheres(scaled_map)
        # scaled_map = self.unnormalize(map_data, field_scale)
        return map_data

    def unnormalize(self, in_map, scale_factors):
        scale = scale_factors['scale']
        out_map = in_map * scale
        return out_map
