import logging
import hydra

from typing import *

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field

from hydra_filesets import (
    DatasetFiles, 
    NoiseSrcFiles, 
    NoiseCacheFiles,
    PlanckInstrumentFiles,
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
def try_fs_locators(cfg):
    logger.debug(f"Running {__name__} in {__file__}")
    try_dataset_files(cfg)
    try_noise_files(cfg)
    try_wmap_files(cfg)
    try_instr_files(cfg)


def try_dataset_files(cfg):
    dataset_files = DatasetFiles(cfg)

    split_dummy0 = dataset_files.get_split("Dummy0")
    this_sim = split_dummy0.get_sim(0)
    print(this_sim.sim_config_path)
    print(this_sim.cmb_map_fid_path)
    print(this_sim.cmb_ps_fid_path)
    print(this_sim.cmb_ps_der_path)
    print(this_sim.obs_map_path(100))

    split_dummy1 = dataset_files.get_split("Dummy1")
    this_sim = split_dummy1.get_sim(0)
    print(this_sim.sim_config_path)
    print(this_sim.cmb_map_fid_path)
    print(this_sim.cmb_ps_fid_path)    # Note that this is different
    print(this_sim.cmb_ps_der_path)
    print(this_sim.obs_map_path(100))


def try_noise_files(cfg):
    noise_src_files = NoiseSrcFiles(cfg)
    path = noise_src_files.get_path_for(100)
    print(path, f"which does {'' if path.exists() else 'not '}exist.")
    
    noise_cache_files = NoiseCacheFiles(cfg)
    path = noise_cache_files.get_path_for(100, "T")
    print(path, f"which does {'' if path.exists() else 'not '}exist.")
    pass


def try_wmap_files(cfg):
    wmap_files = WMAPFiles(cfg)
    print(wmap_files.wmap_chains_dir)


def try_instr_files(cfg):
    instr_files = PlanckInstrumentFiles(cfg)
    print(instr_files.instr_table_path)


if __name__ == "__main__":
    try_fs_locators()
