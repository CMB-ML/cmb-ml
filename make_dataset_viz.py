from typing import List, Dict, Any
import logging
from dataclasses import dataclass, field
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from pysm3.models.template import read_txt as pysm_read_txt

from namer_dataset_output import DatasetFilesNamer, SimFilesNamer
from utils.planck_cmap import colombi1_cmap
from utils.fits_inspection import get_num_fields


logger = logging.getLogger(__name__)


@dataclass
class VizFilesCfg:
    viz_dir: str = "viz"
    viz_cmb_name: str = "cmb_{field}"
    viz_obs_name: str = "obs_{freq}_{field}"
    viz_ps_fid_name: str = "ps_fid_{field}"
    viz_ps_der_name: str = "ps_der_{field}"
    viz_ps_combo_name: str = "ps_combo_{field}"

@dataclass
class LocalConfig:
    defaults: List[Any] = field(default_factory=lambda: [
        "config",
        "_self_"
        ])
    file_system: VizFilesCfg = field(default_factory=VizFilesCfg)

cs = ConfigStore.instance()
cs.store(name="this_config", node=LocalConfig)


@hydra.main(version_base=None, config_path="cfg", config_name="this_config")
def try_cmb_from_conf(cfg):
    logger.debug(f"Running {__name__} in {__file__}")
    
    # Making global setup - the only place I want to pull from the conf
    dataset_files = DatasetFilesNamer(cfg)

    # Visualize maps
    map_maker = MapVizMaker(cfg)
    ps_plot_maker = PSVizMaker(cfg)

    # Look at Dummy0:0 sim only
    split = dataset_files.get_split(0)
    sim = split.get_sim(0)
    
    map_maker.make(sim)
    ps_plot_maker.make(sim)


class VizFilesNamer:
    def __init__(self, cfg):
        self.dir = cfg.file_system.viz_dir
        self.cmb_name = cfg.file_system.viz_cmb_name
        self.obs_name = cfg.file_system.viz_obs_name
        self.ps_fid_name = cfg.file_system.viz_ps_fid_name
        self.ps_der_name = cfg.file_system.viz_ps_der_name
        self.ps_combo_name = cfg.file_system.viz_ps_combo_name

    def cmb_path(self, sim: SimFilesNamer, map_field):
        return self._make_path(sim, self.cmb_name.format(field=map_field))

    def obs_path(self, sim: SimFilesNamer, freq, map_field):
        return self._make_path(sim, self.obs_name.format(freq=freq, field=map_field))

    def ps_fid_path(self, sim: SimFilesNamer, map_field):
        return self._make_path(sim, self.ps_fid_name.format(field=map_field))

    def ps_der_path(self, sim: SimFilesNamer, map_field):
        return self._make_path(sim, self.ps_der_name.format(field=map_field))

    def ps_combo_path(self, sim: SimFilesNamer, map_field):
        return self._make_path(sim, self.ps_combo_name.format(field=map_field))

    def _make_path(self, sim, fn):
        path: Path = sim.path / self.dir / fn
        path.parent.mkdir(exist_ok=True, parents=True)
        return path


class MapVizMaker:
    def __init__(self, cfg) -> None:
        self.namer = VizFilesNamer(cfg)
        self.dsf = DatasetFilesNamer(cfg)
        self.freqs = list(cfg.simulation.detector_freqs)

        self.curr_sim = None
        self.curr_type = None
        self.curr_freq = None
        self.curr_field = None

    def make(self, sim: SimFilesNamer) -> None:
        # Pretend to be at sim level (no dependence on config)
        logger.info(f"Making map vizualizations for {sim.sfl.name}:{sim.sim_num}")
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


class PSVizMaker:
    def __init__(self, cfg):
        self.namer = VizFilesNamer(cfg)
        self.dsf = DatasetFilesNamer(cfg)

    def make(self, sim: SimFilesNamer):
        logger.info(f"Making power spectrum visualizations for {sim.sfl.name}:{sim.sim_num}")
        self.viz_fid_cmb_ps(sim)
        self.viz_der_cmb_ps(sim)
        self.viz_der_and_fid_ps(sim)

    def viz_fid_cmb_ps(self, sim: SimFilesNamer) -> None:
        in_ps_path = sim.cmb_ps_fid_path
        title = "Fiducial Power Spectrum, {field_str}"
        namer = self.namer.ps_fid_path
        self._viz_ps(in_ps_path, sim, title, namer)

    def viz_der_cmb_ps(self, sim:SimFilesNamer) -> None:
        in_ps_path = sim.cmb_ps_der_path
        title = "Derived Power Spectrum, {field_str}"
        namer = self.namer.ps_der_path
        self._viz_ps(in_ps_path, sim, title, namer)

    def _viz_ps(self, in_ps_path, sim:SimFilesNamer, title: str, namer):
        n_fields = self.get_n_fields_ps(in_ps_path)
        ps = self.load_ps(in_ps_path)
        ells = ps[2:, 0]
        field_strs = ["L", "TT", "EE", "BB"]
        for field_idx in range(1, 1 + n_fields):
            this_ps = ps[2:, field_idx]
            field_str = field_strs[field_idx]
            plt.plot(ells, this_ps)
            plt.title(label=title.format(field_str=field_str))
            out_img_path = namer(sim, field_idx)
            plt.savefig(out_img_path)
            plt.close()

    def viz_der_and_fid_ps(self, sim: SimFilesNamer):
        in_ps_paths = [sim.cmb_ps_fid_path, sim.cmb_ps_der_path]
        title = "Both Power Spectra, {field_str}"
        namer = self.namer.ps_combo_path
        self._viz_many_ps(in_ps_paths, sim, title, namer)

    def _viz_many_ps(self, 
                     in_ps_paths: List[np.ndarray], 
                     sim:SimFilesNamer, 
                     title: str, 
                     namer):
        ps_s = []
        n_fields = None
        ell_max = None
        for in_ps_path in in_ps_paths:
            n_fields_in = self.get_n_fields_ps(in_ps_path)
            if n_fields is None:
                n_fields = n_fields_in
            else:
                try: 
                    assert n_fields == n_fields_in
                except AssertionError as e:
                    print(f"Different numbers of fields; unsure how to handle this. Paths with conflict: {in_ps_paths[0]}, {in_ps_path}")
            ps = self.load_ps(in_ps_path)
            if ell_max is None:
                ell_max = ps.shape[0]
            else:
                if ps.shape[0] < ell_max: 
                    ell_max = ps.shape[0]
            ps_s.append(ps)
        ells = ps[2:ell_max, 0]
        field_strs = ["L", "TT", "EE", "BB"]
        del ps
        for field_idx in range(1, 1 + n_fields):
            for ps in ps_s:
                this_ps = ps[2:ell_max, field_idx]
                plt.plot(ells, this_ps)
            field_str = field_strs[field_idx]
            plt.title(label=title.format(field_str=field_str))
            out_img_path = namer(sim, field_str)
            plt.savefig(out_img_path)
            plt.close()

    def get_n_fields_ps(self, in_path: Path):
        # TODO Unbreak
        return 3

    def load_ps(self, in_path: Path) -> np.ndarray:
        return pysm_read_txt(in_path)


if __name__ == "__main__":
    try_cmb_from_conf()
