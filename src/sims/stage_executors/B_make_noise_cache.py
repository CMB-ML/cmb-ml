from typing import Dict
import pysm3
import logging

import hydra
from omegaconf import DictConfig
from pathlib import Path

from ..handlers.noisecache_handler import NoiseCacheHandler # register handler
from ..handlers.qtable_handler import QTableHandler # register handler

from ..detector import make_detector
import utils.fits_inspection as fits_inspect
from ..physics_instrument_noise import planck_result_to_sd_map

from ...core import (
    BaseStageExecutor,
    ExperimentParameters,
    Asset
)

logger = logging.getLogger(__name__)
class NoiseCacheExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig, experiment: ExperimentParameters) -> None:
        # The following stage_str must match the pipeline yaml
        super().__init__(cfg, experiment, stage_str='make_noise_cache')

        self.planck_deltabandpass = self.assets_in['planck_deltabandpass']
        
        with self.name_tracker.set_context('src_root', cfg.local_system.noise_src_dir):
            self.deltabandpass = self.planck_deltabandpass.read()
        
        self.noise_src = cfg.local_system.noise_src_dir
        self.noise_files: Dict[str, str] = dict(cfg.simulation.noise)
        self.out_noise_cache: Asset = self.assets_out['noise_cache']

        
    def execute(self) -> None:
        plank_freqs = self.experiment.detector_freqs
        field_strings = self.experiment.map_fields
        
        nside = self.experiment.nside
        # field_strings = 'TQU'
        for freq in plank_freqs:
            detector = make_detector(self.deltabandpass, freq)
            src_path = self.get_src_path(freq)
            for field_str in field_strings:
                if freq in [545, 857] and field_str != 'T':
                    logger.debug(f"Skipping {freq} GHz, {field_str}: source data not found at frequency.")
                    continue
                hdu = fits_inspect.ASSUME_FITS_HEADER
                field_idx = fits_inspect.lookup_field_idx(field_str, src_path, hdu)
                st_dev_skymap = planck_result_to_sd_map(fits_fn=src_path, hdu=hdu, field_idx=field_idx, nside_out=nside, cen_freq=detector.cen_freq)
                with self.name_tracker.set_context('freq', freq):
                    with self.name_tracker.set_context('field', field_str):
                    
                        logger.debug(f'writing to path: {self.out_noise_cache.path}')
                        self.out_noise_cache.write(st_dev_skymap, field_str)
                        logger.debug(f'wrote to path: {self.out_noise_cache.path}')
                    
    def get_src_path(self, detector):
        #TODO: add detector checks for existence in file
        det_key = f'det{detector:03d}'
        fn = self.noise_files[det_key]
        return Path(self.noise_src) / fn

    
    


