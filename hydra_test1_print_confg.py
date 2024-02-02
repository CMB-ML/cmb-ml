import logging
import hydra
from omegaconf import OmegaConf


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def print_configs(cfg):
    logger.setLevel("DEBUG")
    print(OmegaConf.to_yaml(cfg))
    pass


if __name__ == "__main__":
    print_configs()
