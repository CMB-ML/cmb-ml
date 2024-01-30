from typing import *
import logging
from dataclasses import dataclass, field
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore

from astropy.io import fits

import healpy as hp
import matplotlib.pyplot as plt

from utils.hydra_log_helper import *
from hydra_filesets import DatasetFiles, SimFiles
# from component_seed_maker import SimLevelSeedMaker, FieldLevelSeedMaker
from planck_instrument import (
    PlanckInstrument, 
    make_planck_instrument )
from planck_cmap import colombi1_cmap
from inspect_fits_file import get_num_fields
# from component_cmb import CMBMaker, make_cmb_maker, save_der_cmb_ps, save_fid_cmb_map

# Goal: Use a conf to make the CMB component


logger = logging.getLogger(__name__)


@dataclass
class SplitsDummy:
    ps_fidu_fixed: bool = False
    n_sims: int = 1

@dataclass
class VizFileSystem:
    viz_dir: str = "viz"
    viz_cmb_name: str = "cmb_{field}"
    viz_obs_name: str = "obs_{freq}_{field}"

@dataclass
class DummyConfig:
    defaults: List[Any] = field(default_factory=lambda: [
        "config_no_out",
        "_self_"
        ])
    splits: Dict[str, SplitsDummy] = field(default_factory=lambda: {
        "Dummy0": SplitsDummy,
        # "Dummy1": SplitsDummy(ps_fidu_fixed=True)
    })
    file_system: VizFileSystem = field(default_factory=VizFileSystem)
    dataset_name: str = "Dummy_CMB_Map_Check"

cs = ConfigStore.instance()
cs.store(name="this_config", node=DummyConfig)


@hydra.main(version_base=None, config_path="cfg", config_name="this_config")
def try_cmb_from_conf(cfg):
    logger.debug(f"Running {__name__} in {__file__}")
    
    # Making global setup - the only place I want to pull from the conf
    dataset_files = DatasetFiles(cfg)

    # Visualize maps
    viz_maker = VizMaker(cfg)
    # viz_namer = VizNamer(cfg)

    # Check power spectra

    # planck_freqs = list(cfg.simulation.detector_freqs)

    # # Pretend to be at sim level (no dependence on config)
    split = dataset_files.get_split("Dummy0")
    sim = split.get_sim(0)
    viz_maker.make(sim)

    # examine_cmb_map(sim, viz_namer)
    # examine_obs_maps(sim, planck_freqs, viz_namer)


class VizMaker:
    def __init__(self, cfg) -> None:
        self.namer = VizFiles(cfg)
        self.dsf = DatasetFiles(cfg)
        self.freqs = list(cfg.simulation.detector_freqs)

        self.curr_sim = None
        self.curr_type = None
        self.curr_freq = None
        self.curr_field = None

    def make(self, sim: SimFiles) -> None:
        # Pretend to be at sim level (no dependence on config)
        logger.info(f"Making vizualizations for {sim.sfl.name}:{sim.sim_num}")
        self.curr_sim = sim
        self.viz_cmb_map()
        self.viz_obs_maps()
        self.curr_sim = None
    
    def viz_cmb_map(self) -> None:
        logger.info(f"Making vizualization  for cmb.")
        self.curr_type = "CMB"
        self.viz_map()
        self.curr_type = None

    def viz_obs_maps(self) -> None:
        self.curr_type = "Observation"
        for nom_freq in self.freqs:
            logger.info(f"Making vizualization  for obs at {nom_freq}.")
            self.curr_freq = nom_freq
            self.viz_map()
            self.curr_freq = None
        self.curr_type = None
    
    def viz_map(self) -> None:
        fields_strs = "TQU"
        n_fields = get_num_fields(self.in_map_path)[1]  # Assume HDUL 1, which is valid because all maps are produced by us.
        for field_idx in range(n_fields):
            self.curr_field = fields_strs[field_idx]
            this_map = hp.read_map(self.in_map_path, field=field_idx)
            title = self.map_title
            hp.mollview(this_map, title=title, cmap=colombi1_cmap)
            
            plt.savefig(self.out_img_path)
            plt.close()
            self.curr_field = None

    @property
    def map_title(self) -> str:
        if self.curr_freq is not None:
            freq_str = f" at {self.curr_freq}"
        else:
            freq_str = ""
        field_str = f", {self.curr_field} Field"
        return f"{self.curr_type}{freq_str}{field_str}"
    
    @property
    def in_map_path(self) -> Path:
        if self.curr_type == "CMB":
            path = self.curr_sim.cmb_map_fid_path
        else:
            path = self.curr_sim.obs_map_path(self.curr_freq)
        return path

    @property
    def out_img_path(self) -> Path:
        if self.curr_type == "CMB":
            path = self.namer.cmb_path(self.curr_sim, self.curr_field)
        else:
            path = self.namer.obs_path(self.curr_sim, self.curr_freq, self.curr_field)
        return path


class VizFiles:
    def __init__(self, cfg):
        self.dir = cfg.file_system.viz_dir
        self.cmb_name = cfg.file_system.viz_cmb_name
        self.obs_name_template = cfg.file_system.viz_obs_name

    def cmb_path(self, sim: SimFiles, map_field):
        path:Path = sim.path / self.dir / self.cmb_name.format(field=map_field)
        path.parent.mkdir(exist_ok=True, parents=True)
        return path

    def obs_path(self, sim: SimFiles, freq, map_field):
        path:Path = sim.path / self.dir / self.obs_name_template.format(freq=freq, field=map_field)
        path.parent.mkdir(exist_ok=True, parents=True)
        return path


if __name__ == "__main__":
    try_cmb_from_conf()
