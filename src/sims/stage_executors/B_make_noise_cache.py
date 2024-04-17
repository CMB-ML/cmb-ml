from typing import Dict
import pysm3

import hydra
from omegaconf import DictConfig
from pathlib import Path

from astropy.table import QTable

from .noise_cache import NoiseCacheCreator, Detector

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

class NoiseCacheExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig, experiment: ExperimentParameters) -> None:
        self.stage_str = 'create-noise'
        super().__init__(cfg, experiment)

        self.instr_table_path = Path(cfg.local_system.instr_table_path)
        self.table = self.instr_table_path

        self.noise_src = cfg.local_system.noise_src_dir
        self.noise_files: Dict[str, str] = dict(cfg.simulation.noise)
        self.out_noise_cache: Asset = self.assets_out['noise_cache']

        # self.planck: Instrument = make_planck_instrument(cfg)

        # self.noise_factory: InstrumentNoiseFactory = make_noise_maker(cfg, self.planck)
        # self.noise_seed_factory = FieldLevelSeedFactory(cfg, "noise")
        # self.noise: InstrumentNoise = self.noise_factory.make_instrument_noise()
        

    def execute(self) -> None:
        plank_freqs = self.experiment.detector_freqs
        field_strings = self.experiment.map_fields
        
        nside = self.experiment.nside
        # field_strings = 'TQU'
        for freq in plank_freqs:
            detector = self.make_detector(freq)
            src_path = self.get_src_path(freq)
            for field_str in field_strings:
                cache_creator = NoiseCacheCreator(src_path, nside, detector.cen_freq)
                with self.name_tracker.set_context('freq', freq):
                    with self.name_tracker.set_context('field', field_str):
                        with self.name_tracker.set_context('nside', nside):
                            print('writing to path:', self.out_noise_cache.path)
                            cache_creator.create_noise_cache(field_str, self.out_noise_cache.path)
                           
                
    
    def process_split(self, split: Split) -> None:
        pass


    def get_src_path(self, detector):
        det_key = f'det{detector:03d}'
        fn = self.noise_files[det_key]
        return Path(self.noise_src) / fn
    
    def read_instr_table(self) -> QTable:
        planck_beam_info = QTable.read(self.table, format="ascii.ipac")
        planck_beam_info.add_index("band")
        return planck_beam_info

    def make_detector(self, band):
        table = self.read_instr_table()
        
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
    
    


