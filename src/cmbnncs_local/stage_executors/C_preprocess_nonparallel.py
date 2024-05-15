from typing import List, Dict
import logging

import numpy as np

from omegaconf import DictConfig
from tqdm import tqdm

from core import (
    BaseStageExecutor, 
    Split,
    Asset
    )
from cmbnncs.spherical import sphere2piecePlane
from ..handler_npymap import NumpyMap             # Import to register the AssetHandler
from core.asset_handlers.asset_handlers_base import Config # Import for typing hint
from core.asset_handlers.healpy_map_handler import HealpyMap # Import for typing hint
from src.utils import make_instrument, Instrument, Detector


logger = logging.getLogger(__name__)


class NonParallelPreprocessExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="preprocess")

        self.instrument: Instrument = make_instrument(cfg=cfg)

        self.out_cmb_asset: Asset = self.assets_out["cmb_map"]
        self.out_obs_assets: Asset = self.assets_out["obs_maps"]
        out_cmb_map_handler: NumpyMap
        out_obs_map_handler: NumpyMap

        self.in_norm_file: Asset = self.assets_in["norm_file"]
        self.in_cmb_asset: Asset = self.assets_in["cmb_map"]
        self.in_obs_assets: Asset = self.assets_in["obs_maps"]
        in_norm_file_handler: Config
        in_cmb_map_handler: HealpyMap
        in_obs_map_handler: HealpyMap

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        self.default_execute()

    def process_split(self, 
                      split: Split) -> None:
        logger.info(f"Running {self.__class__.__name__} process_split() for split: {split.name}.")
        logger.debug(f"Reading norm_file from: {self.in_norm_file.path}")
        scale_factors = self.in_norm_file.read()
        for sim in tqdm(split.iter_sims(), total=split.n_sims):
            with self.name_tracker.set_context("sim_num", sim):
                self.process_sim(scale_factors)

    def process_sim(self, scale_factors) -> None:
        in_cmb_map = self.in_cmb_asset.read()
        scaled_map = self.process_map(in_cmb_map, 
                                      scale_factors=scale_factors['cmb'], 
                                      detector_fields=self.cfg.scenario.map_fields)
        self.out_cmb_asset.write(data=scaled_map)

        for freq, detector in self.instrument.dets.items():
            with self.name_tracker.set_context('freq', freq):
                obs_map = self.in_obs_assets.read()
                scaled_map = self.process_map(obs_map,
                                              scale_factors=scale_factors[freq],
                                              detector_fields=detector.fields)
                self.out_obs_assets.write(data=scaled_map)

    def process_map(self, 
                    map_data: np.ndarray, 
                    scale_factors: Dict[str, Dict[str, float]],
                    detector_fields: str
                    ) -> List[np.ndarray]:
        processed_maps = []
        all_fields:str = self.cfg.scenario.map_fields  # Either I or IQU
        for field_char in detector_fields:
            field_idx = all_fields.find(field_char)
            field_scale = scale_factors[field_char]
            field_data = map_data[field_idx]
            scaled_map = self.normalize(field_data, field_scale)
            mangled_map = sphere2piecePlane(scaled_map)
            processed_maps.append(mangled_map)
        return processed_maps

    def normalize(self, in_map, scale_factors):
        scale = scale_factors['scale']
        out_map = in_map / scale
        return out_map
