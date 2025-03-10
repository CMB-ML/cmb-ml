from typing import List, Dict, Union
import logging

import numpy as np

from omegaconf import DictConfig
from tqdm import tqdm
import pysm3.units as u

from cmbml.core import (
    BaseStageExecutor, 
    Split,
    Asset
    )
from cmbnncs.spherical import sphere2piecePlane
from cmbml.core.asset_handlers import (
    QTableHandler,
    NumpyMap,
    HealpyMap,
    Config
    )
from cmbml.utils import make_instrument, Instrument, Detector


logger = logging.getLogger(__name__)


class NonParallelPreprocessExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="preprocess")

        self.out_cmb_asset: Asset = self.assets_out["cmb_map"]
        self.out_obs_assets: Asset = self.assets_out["obs_maps"]
        out_cmb_map_handler: NumpyMap
        out_obs_map_handler: NumpyMap

        # self.in_dataset_stats: Asset = self.assets_in["dataset_stats"]
        self.in_cmb_asset: Asset = self.assets_in["cmb_map"]
        self.in_obs_assets: Asset = self.assets_in["obs_maps"]
        in_det_table: Asset  = self.assets_in['deltabandpass']
        # in_dataset_stats_handler: Config
        in_cmb_map_handler: HealpyMap
        in_obs_map_handler: HealpyMap
        in_det_table_handler: QTableHandler

        det_info = in_det_table.read()
        self.instrument: Instrument = make_instrument(cfg=cfg, det_info=det_info)

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        self.default_execute()

    def process_split(self, 
                      split: Split) -> None:
        logger.info(f"Running {self.__class__.__name__} process_split() for split: {split.name}.")
        # logger.debug(f"Reading dataset_stats from: {self.in_dataset_stats.path}")
        # scale_factors = self.in_dataset_stats.read()
        for sim in tqdm(split.iter_sims(), total=split.n_sims):
            with self.name_tracker.set_context("sim_num", sim):
                self.process_sim()
                # self.process_sim(scale_factors)

    def process_sim(self, 
                    # scale_factors
                    ) -> None:
        in_cmb_map = self.in_cmb_asset.read()
        scaled_map = self.process_map(in_cmb_map, 
                                    #   scale_factors=scale_factors['cmb'], 
                                      detector='cmb')
        self.out_cmb_asset.write(data=scaled_map)

        for freq, detector in self.instrument.dets.items():
            with self.name_tracker.set_context('freq', freq):
                obs_map = self.in_obs_assets.read()
                scaled_map = self.process_map(obs_map,
                                            #   scale_factors=scale_factors[freq],
                                              detector=detector)
                self.out_obs_assets.write(data=scaled_map)

    def process_map(self,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
                    map_data: u.Quantity, 
                    # scale_factors: Dict[str, Dict[str, float]],
                    detector: Union[Detector, str]
                    ) -> List[np.ndarray]:
        if detector == 'cmb':
            detector_fields = self.cfg.scenario.map_fields
            equivalencies = None
        else:
            detector_fields = detector.fields
            equivalencies = u.cmb_equivalencies(detector.cen_freq)
        processed_maps = []
        all_fields:str = self.cfg.scenario.map_fields  # Either I or IQU
        for field_char in detector_fields:
            field_idx = all_fields.find(field_char)  # We assume these files were created by this library
            field_data = map_data[field_idx]
            # field_scale = scale_factors[field_char]
            # scaled_map = self.apply_scale(field_data, field_scale)
            # scaled_map = scaled_map.to_value(u.uK_CMB, equivalencies=equivalencies)
            # mangled_map = sphere2piecePlane(scaled_map)
            scaled_map = field_data.to_value(u.uK_CMB, equivalencies=equivalencies)
            mangled_map = sphere2piecePlane(scaled_map)
            processed_maps.append(mangled_map)
        return processed_maps

    # def apply_scale(self, in_map, scale_factors):
    #     scale = scale_factors['scale']
    #     out_map = in_map / scale
    #     return out_map
