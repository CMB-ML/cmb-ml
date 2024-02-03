from typing import List, Dict
import logging
from pathlib import Path

from omegaconf import DictConfig


logger = logging.getLogger(__name__)


class WMAPFilesNamer:
    def __init__(self, conf: DictConfig) -> None:
        logger.debug(f"Running {__name__} in {__file__}")
        self.wmap_chains_dir = Path(conf.local_system.wmap_chains_dir)
