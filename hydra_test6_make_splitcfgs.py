import logging
import hydra

from typing import *

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field

from utils.hydra_log_helper import *
from hydra_filesets import (
    DatasetFiles, 
    NoiseSrcFiles, 
    NoiseCacheFiles,
    InstrumentFiles,
    WMAPFiles
    )


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
def try_make_split_configs(cfg):
    logger.debug(f"Running {__name__} in {__file__}")
    # print(OmegaConf.to_yaml(cfg))

    dataset_files = DatasetFiles(cfg)
    check_dataset_root_exists(dataset_files)
    make_folder_for_each_split(dataset_files)
    pass


def check_dataset_root_exists(dataset_files: DatasetFiles):
    try:
        dataset_files.assume_dataset_root_exists()
    except Exception as e:
        print(e)
        exit()


def make_folder_for_each_split(dataset_files: DatasetFiles):
    for split in dataset_files.iter_splits():
        for sim in split.iter_sims():
            sim.make_folder()


def make_split_configs(dataset_files: DatasetFiles):
    for split in dataset_files.iter_splits():
        pass
        # Need to write only the following (I think?)
        # Only interesting bit is the wmap_chain_idcs ???
        """
ps_fidu_fixed: true
n_sims: 2
wmap_chain_idcs: 1, 3
"""


if __name__ == "__main__":
    try_make_split_configs()
    pass
