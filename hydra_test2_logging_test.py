import logging
import hydra
from hydra_test2_logging_testhelper import DatasetFilepaths


logger = logging.getLogger(__name__)
"""
Levels are: 
NOTSET    0
DEBUG    10
INFO     20
WARNING  30
ERROR    40
CRITICAL 50
"""


# TODO: Decide if I want to be able to change logger level within Python code. If so, change hydra config or establish correct method.


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def test_class(cfg):
    logger.debug(f"Running {__name__} in {__file__}")
    
    logger.debug(f"Debug level in {__name__}")
    logger.info(f"Info level in {__name__}")
    logger.warning(f"Warning level in {__name__}")
    
    a = DatasetFilepaths(cfg)
    
    # Check if DatasetFilepaths logger affects __main__ logger (no)
    logger.debug(f"Debug level in {__name__}")
    logger.info(f"Info level in {__name__}")
    logger.warning(f"Warning level in {__name__}")

    # Check if setLevel affects things (currently, yes)
    logger.setLevel("DEBUG")
    logger.debug(f"Debug level in {__name__}, after setLevel DEBUG")
    logger.info(f"Info level in {__name__}, after setLevel DEBUG")
    logger.warning(f"Warning level in {__name__}, after setLevel DEBUG")
    pass


if __name__ == "__main__":
    test_class()
