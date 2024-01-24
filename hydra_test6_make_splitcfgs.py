from typing import *
import logging
from dataclasses import dataclass, field

import numpy as np

import hydra
from hydra.core.config_store import ConfigStore

from utils.hydra_log_helper import *
from hydra_filesets import (
    DatasetFiles,
    NoiseSrcFiles, 
    NoiseCacheFiles,
    InstrumentFiles,
    WMAPFiles,
    DatasetConfigsBuilder
    )


logger = logging.getLogger(__name__)


@dataclass
class SplitsDummy:
    ps_fidu_fixed: bool = False
    n_sims: int = 4

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
    dataset_configs_builder = DatasetConfigsBuilder(dataset_files)
    dataset_files.assume_dataset_root_exists()
    dataset_configs_builder.setup_folders()

    rng = np.random.default_rng(seed=8675309)
    chain_idcs_dict = dataset_configs_builder.make_chain_idcs_per_split(rng)
    dataset_configs_builder.make_split_configs(chain_idcs_dict)


if __name__ == "__main__":
    try_make_split_configs()

