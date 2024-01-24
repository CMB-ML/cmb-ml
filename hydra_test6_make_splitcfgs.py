import random
from typing import *

import logging
import hydra

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
    check_dataset_root_exists(dataset_files)
    make_folder_for_each_split(dataset_files)
    make_split_configs(dataset_files)
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
    chain_rows = 1000  # TODO: Get correct value... in configs? Hard code it?
    
    n_indices_total = dataset_files.total_n_ps
    all_chain_indices = [random.randint(0, chain_rows) for _ in range(n_indices_total)]
    
    n_indices_used = 0
    for split in dataset_files.iter_splits():
        n_indices_this_split = split.n_ps
        first_index = n_indices_used
        last_index = first_index + n_indices_this_split

        chain_idcs = all_chain_indices[first_index: last_index]
        n_indices_used += n_indices_this_split

        split_cfg_dict = dict(
            ps_fidu_fixed = split.ps_fidu_fixed,
            n_sims = split.n_sims,
            wmap_chain_idcs = chain_idcs
        )

        split_yaml = OmegaConf.to_yaml(OmegaConf.create(split_cfg_dict))
        split.write_yaml_to_conf(split_yaml)


def determine_num_vals(dataset_files: DatasetFiles):
    num_indcs_needed = 0
    for split in dataset_files.iter_splits():
        num_indcs_needed += split.n_ps
    return num_indcs_needed    


if __name__ == "__main__":
    try_make_split_configs()
    pass
