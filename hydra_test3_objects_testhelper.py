from pathlib import Path
from omegaconf import DictConfig
import hydra
import logging


logger = logging.getLogger(__name__)


class FileLocator:
    def __init__(self, create_missing_dest: bool=True) -> None:
        self.create_missing_dest = create_missing_dest

    @staticmethod
    def src_exists(p: Path) -> bool:
        res = False
        try:
            assert p.exists()
            res = True
        except AttributeError as e:
            logger.error(f"References to locations in the file system should be pathlib Paths. Found {p}, type {type(p)}")
            raise e
        except AssertionError:
            logger.error(f"Cannot find {p}")
            raise FileNotFoundError(f"Cannot find {p}")
        return res
    
    def dest_dir_exists(self, p: Path):
        res = False
        try:
            assert p.exists()
            res = True
        except AttributeError as e:
            logger.error(f"References to locations in the file system should be pathlib Paths. Found {p}, type {type(p)}")
            raise e            
        except AssertionError as e:
            if self.create_missing_dest:
                p.mkdir(parents=True, exist_ok=True)
            else:
                logger.error(f"Cannot find {p}")
                raise FileNotFoundError(f"Cannot find {p}")
        return res


class DatasetFileLocator(FileLocator):
    def __init__(self, conf: DictConfig, create_missing_dest: bool=True) -> None:
        logger.info(f"Running {__name__} in {__file__}")
        super().__init__(create_missing_dest)
        self.dataset_root = Path(conf.local_system.datasets_root)
        self.dest_dir_exists(self.dataset_root)



class AssetFileLocator(FileLocator):
    def __init__(self, conf: DictConfig) -> None:
        logger.info(f"Running {__name__} in {__file__}")
        super().__init__(create_missing_dest=False)
        self.dataset_root = Path(conf.local_system.datasets_root)
        self.dest_exists(self.dataset_root)

