from pathlib import Path
from omegaconf import DictConfig
import hydra
import logging


logger = logging.getLogger(__name__)


class DatasetFilepaths:
    def __init__(self, conf: DictConfig):
        logger.debug(f"Running {__name__} in {__file__}")
        # logger.debug(f"Debug level in {__name__}")
        # logger.info(f"Info level in {__name__}")
        # logger.warning(f"Warning level in {__name__}")
        
        # # Check if setLevel affects things (it doesn't with current hydra settings)
        # logger.setLevel("DEBUG")
        # logger.debug(f"Debug level in {__name__}, after setLevel DEBUG")
        # logger.info(f"Info level in {__name__}, after setLevel DEBUG")
        # logger.warning(f"Warning level in {__name__}, after setLevel DEBUG")
        
        self.root = Path(conf.local_system.datasets_root)

    @staticmethod
    def src_exists(p: Path):
        try:
            assert p.exists()
        except AssertionError as e:
            logger.warning(f"Cannot find {p}")
