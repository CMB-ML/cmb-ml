from typing import List, Dict
import logging

import numpy as np

from omegaconf import DictConfig
from tqdm import tqdm
from cmbnncs.spherical import piecePlanes2spheres

from cmbml.core import (
    BaseStageExecutor, 
    Split,
    Asset
    )
from cmbml.utils import make_instrument, Instrument, Detector
from cmbml.core.asset_handlers import (
    NumpyMap,
    Config,
    HealpyMap
    )

logger = logging.getLogger(__name__)


class NonParallelPostprocessExecutor(BaseStageExecutor):
    def __init__(self,
                 cfg: DictConfig) -> None:
        logger.debug("Initializing CMBNNCS PostprocessExecutor")
        super().__init__(cfg, stage_str="postprocess")

        self.instrument: Instrument = make_instrument(cfg=cfg)

        self.out_cmb_asset: Asset = self.assets_out["cmb_map"]
        out_cmb_map_handler: HealpyMap

        self.in_norm_file: Asset = self.assets_in["norm_file"]
        self.in_cmb_asset: Asset = self.assets_in["cmb_map"]
        in_norm_file_handler: Config
        in_cmb_map_handler: NumpyMap

        # self.map_fields = cfg.scenario.map_fields
        # self.model_epochs = cfg.training.postprocess.epoch

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        self.default_execute()

    def process_split(self, 
                      split: Split) -> None:
        logger.info(f"Running {self.__class__.__name__} process_split() for split: {split.name}.")
        # At some point, TODO: implement this for varied epochs
        # logger.info(f"Running {self.__class__.__name__} process_split() for epoch: {self.name_tracker.context['epoch']}, split: {split.name}.")
        logger.debug(f"Reading norm_file from: {self.in_norm_file.path}")
        scale_factors = self.in_norm_file.read()
        for sim in tqdm(split.iter_sims()):
            with self.name_tracker.set_context("sim_num", sim):
                self.process_sim(scale_factors)

    def process_sim(self, scale_factors) -> None:
        # Compare to preprocess_nonparallel; no need to deal with detectors, only cmb map
        for epoch in self.model_epochs:
            with self.name_tracker.set_context("epoch", epoch):
                in_cmb_map = self.in_cmb_asset.read()
                scaled_map = self.unscale_map_file(in_cmb_map, scale_factors=scale_factors['cmb'])
                self.out_cmb_asset.write(scaled_map, 
                                        column_names=[f"STOKES_{f}" for f in self.map_fields],
                                        column_units=["uK_CMB" for _ in self.map_fields])

    def unscale_map_file(self, 
                             map_data: np.ndarray, 
                             scale_factors: Dict[str, Dict[str, float]]) -> List[np.ndarray]:
        processed_maps = [None] * len(self.map_fields)
        for field_char in self.map_fields:
            field_idx = self.map_fields.find(field_char)
            field_scale = scale_factors[field_char]
            field_data = map_data[field_idx]
            scaled_map = self.remove_scale(field_data, field_scale)
            demangled_map = piecePlanes2spheres(scaled_map)
            processed_maps[field_idx] = demangled_map
        return processed_maps

    def remove_scale(self, in_map, scale_factors):
        scale = scale_factors['scale']
        out_map = in_map * scale
        return out_map
