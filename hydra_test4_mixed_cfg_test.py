import logging
import hydra
from typing import List, Dict, Any
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from omegaconf import OmegaConf


logger = logging.getLogger(__name__)


@dataclass
class SplitsDummy:
    ps_fidu_fixed: bool = False
    n_sims: int = 2

@dataclass
class DummyConfig:
    defaults: List[Any] = field(default_factory=lambda: [
        "config_no_out",
        "_self_"
        ])
    splits: Dict[str, SplitsDummy] = field(default_factory=lambda: {
        "Dummy0": SplitsDummy,
        "Dummy1": SplitsDummy(ps_fidu_fixed=True)
    })
    dataset_name: str = "Dummy"

cs = ConfigStore.instance()
cs.store(name="this_config", node=DummyConfig)


@hydra.main(version_base=None, config_path="cfg", config_name="this_config")
def test_class(cfg):
    logger.debug(f"Running in {__name__}")
    print(OmegaConf.to_yaml(cfg))
    pass


if __name__ == "__main__":
    test_class()
