import logging
import hydra
from utils.hydra_log_helper import *


logger = logging.getLogger(__name__)
print("main", __name__)


def log_test():
    logger.info("Info level message.")
    logger.debug("Debug level message.")


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def test_class(cfg):
    logger.setLevel("DEBUG")
    # log_test()
    # print(cfg)
    print_cfg(cfg, caller_name="main")
    pass


if __name__ == "__main__":
    test_class()
