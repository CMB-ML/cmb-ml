import logging
import hydra
from typing import *
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from omegaconf import OmegaConf


logger = logging.getLogger(__name__)


@dataclass
class LocalSystemDummy:
    datasets_root: str = "dummystring"

@dataclass
class FileSystemDummy:
    structure: str = "{dataset_name}/Raw/{split_name}/{sim_folder}"
    sim_folder_prefix: str = "sim"
    sim_str_num_digits: str = 4
    cmb_map_fid_fn: str = "dummyname"
    cmb_ps_fid_fn: str = "dummyname"
    cmb_ps_der_fn: str = "dummyname"
    obs_map_fn: str = "dummyname{det}"
    sim_config_fn: str = "dummyname"

@dataclass
class SplitsDummy:
    ps_fidu_fixed: bool = False
    n_sims: int = 2

@dataclass
class SimulationDummy:
    detector_freqs: List[int] = field(default_factory=lambda: [30, 100])

@dataclass
class DummyConfig:
    local_system: LocalSystemDummy = field(default_factory=LocalSystemDummy)
    file_system: FileSystemDummy = field(default_factory=FileSystemDummy)
    splits: Dict[str, SplitsDummy] = field(default_factory=lambda: {
        "Dummy0": SplitsDummy,
        "Dummy1": SplitsDummy(ps_fidu_fixed=True)
    })
    simulation: SimulationDummy = field(default_factory=SimulationDummy)
    create_dirs: bool = False
    dataset_name: str = "Dummy"

cs = ConfigStore.instance()
cs.store(name="this_config", node=DummyConfig)
cs.store(group="splits", name="Dummy0", node=SplitsDummy)
cs.store(group="splits", name="Dummy1", node=SplitsDummy)

@hydra.main(version_base=None, config_name="this_config")
def test_class(cfg):
    logger.debug(f"Running in {__name__}")
    print(OmegaConf.to_yaml(cfg))
    pass


if __name__ == "__main__":
    test_class()
