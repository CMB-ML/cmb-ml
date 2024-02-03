from typing import List, Dict
import logging
from pathlib import Path

from omegaconf import DictConfig
from astropy.table import QTable
from namer_noise import NoiseCacheFilesNamer, NoiseSrcFilesNamer


logger = logging.getLogger(__name__)


class InstrumentFilesNamer:
    def __init__(self, conf: DictConfig) -> None:
        logger.debug(f"Running {__name__} in {__file__}")
        self.instr_table_path = Path(conf.local_system.instr_table_path)
        self.table = self.instr_table_path
        self.noise_src_files = NoiseSrcFilesNamer(conf)
        self.src = self.noise_src_files
        self.noise_cache_files = NoiseCacheFilesNamer(conf)
        self.cache = self.noise_cache_files

    def read_instr_table(self) -> QTable:
        planck_beam_info = QTable.read(self.table, format="ascii.ipac")
        planck_beam_info.add_index("band")
        return planck_beam_info
