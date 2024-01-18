import logging
import hydra
from utils.hydra_log_helper import *
from hydra_test3_objects_testhelper import DatasetFilepaths


logger = logging.getLogger(__name__)


# TODO: Placeholder only


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def test_class(cfg):
    logger.debug(f"Running {__name__} in {__file__}")
        
    a = DatasetFilepaths(cfg)
    
    pass


if __name__ == "__main__":
    test_class()
