import logging
import hydra

from typing import *

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field

from utils.hydra_log_helper import *
from hydra_test5_objects_testhelper import DatasetFileLocator


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
    logger.debug(f"Running {__name__} in {__file__}")
    # print(OmegaConf.to_yaml(cfg))
    dfl = DatasetFileLocator(cfg)

    sfl_dummy0 = dfl.get_split("Dummy0")
    this_sim = sfl_dummy0.get_sim(0)
    print(this_sim.cmb_map_fid_path)
    print(this_sim.cmb_ps_fid_path)
    print(this_sim.cmb_ps_der_path)
    print(this_sim.obs_map_path(100))

    sfl_dummy1 = dfl.get_split("Dummy1")
    this_sim = sfl_dummy1.get_sim(0)
    print(this_sim.cmb_map_fid_path)
    print(this_sim.cmb_ps_fid_path)    # Note that this is different
    print(this_sim.cmb_ps_der_path)
    print(this_sim.obs_map_path(100))
    pass


if __name__ == "__main__":
    test_class()
