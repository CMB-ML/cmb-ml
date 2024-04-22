from typing import Dict
import pysm3
import logging

import hydra
from omegaconf import DictConfig
from pathlib import Path

from astropy.table import QTable

from .noise_cache import Detector
import utils.fits_inspection as fits_inspect
from ..physics_instrument_noise import planck_result_to_sd_map


# from ..component_instrument_noise import NoiseCacheCreator
# from ..component_seed_maker import (SimLevelSeedFactory,
#                                   FieldLevelSeedFactory)
# from ..component_instrument import Instrument, make_planck_instrument
# from ..component_instrument_noise import (InstrumentNoise, 
#                                         InstrumentNoiseFactory, 
#                                         make_noise_maker)

from ...core import (
    BaseStageExecutor,
    ExperimentParameters,
    Split,
    Asset
)

logger = logging.getLogger(__name__)
class NoiseCacheExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig, experiment: ExperimentParameters) -> None:
        self.stage_str = 'create-noise'
        super().__init__(cfg, experiment)

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
            detector = self.make_detector(freq)
            src_path = self.get_src_path(freq)
            for field_str in field_strings:
                hdu = fits_inspect.ASSUME_FITS_HEADER
                field_idx = fits_inspect.lookup_field_idx(field_str, src_path, hdu)
                st_dev_skymap = planck_result_to_sd_map(fits_fn=src_path, hdu=hdu, field_idx=field_idx, nside_out=nside, cen_freq=detector.cen_freq)
                with self.name_tracker.set_context('freq', freq):
                    with self.name_tracker.set_context('field', field_str):
                    
                        logger.debug(f'writing to path: {self.out_noise_cache.path}')
                        self.out_noise_cache.write(st_dev_skymap, field_str)
                        logger.debug(f'wrote to path: {self.out_noise_cache.path}')
                    
    def get_src_path(self, detector):
        det_key = f'det{detector:03d}'
        fn = self.noise_files[det_key]
        return Path(self.noise_src) / fn
    
    def read_instr_table(self) -> QTable:
        planck_beam_info = QTable.read(self.table, format="ascii.ipac")
        planck_beam_info.add_index("band")
        return planck_beam_info

    def make_detector(self, band):
        # table = self.read_instr_table()
        table = self.deltabandpass
        
        band_str = str(band)
        try:
            assert band_str in table['band']
        except AssertionError:
            raise KeyError(f"A detector specified in the configs, {band} " \
                            f"(converted to {band_str}) does not exist in " \
                            f"the QTable ({self.table}).")
        
        center_frequency = table.loc[band_str]['center_frequency']
        fwhm = table.loc[band_str]['fwhm']
        return Detector(nom_freq=band, cen_freq=center_frequency, fwhm=fwhm)
    
    


