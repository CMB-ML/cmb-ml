from pathlib import Path
from typing import List, Dict, Any, Union
import logging

import numpy as np

import healpy as hp

from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger(__name__)


# TODO: Refactor: unsure of exact structure.
#       Something to track just file locations, something else to handle io
#       Something(s) to track simulation parameters.


class DatasetFilesNamer:
    def __init__(self, conf: DictConfig) -> None:
        logger.debug(f"Running {self.__class__.__name__} in {__name__}")
        self.name: str = conf.dataset_name
        self.root = Path(conf.local_system.datasets_root)
        dir_str_template = conf.file_system.dir_to_dataset
        dir_str = dir_str_template.format(dataset_name=self.name)
        self.path = self.root / dir_str
        self.freqs: List[int] = conf.simulation.detector_freqs

        # For use in SplitFolderLocator
        self.split_structures: Dict[str, Any] = dict(conf.splits)
        self.split_cfg_fn: str = conf.file_system.split_cfg_fn
        self.dir_to_split_template: str = conf.file_system.dir_to_split

        # For use in SimFolderLocator
        self.sim_folder_prefix: str = conf.file_system.sim_folder_prefix
        self.sim_str_num_digits: int = conf.file_system.sim_str_num_digits
        self.dir_to_sim_template: str = conf.file_system.dir_to_sim

        self.sim_config_fn: str = conf.file_system.sim_config_fn
        self.cmb_map_fid_fn: str = conf.file_system.cmb_map_fid_fn
        self.cmb_ps_der_fn: str = conf.file_system.cmb_ps_der_fn
        self.obs_map_fn: str = conf.file_system.obs_map_fn

        # For use in either split or sim level, depending on fixed_fidu
        self.cmb_ps_fid_fn: str = conf.file_system.cmb_ps_fid_fn
        self.cosmo_param_fn: str = conf.file_system.cosmo_param_fn

    def assume_dataset_root_exists(self) -> None:
        try:
            assert self.root.exists()
        except:
            raise FileNotFoundError(f"Dataset root directory does not exist: {self.root}")

    def get_split(self, split_id: Union[str, int]):
        if isinstance(split_id, int):
            try: 
                split_str = list(self.split_structures.keys())[split_id]
            except IndexError:
                n_splits = len(self.split_structures.keys())
                raise ValueError(f"Configuration files specify only {n_splits} keys; {split_id} requested.")
        elif isinstance(split_id, str):
            try: 
                assert split_id in self.split_structures.keys()
                split_str = split_id
            except AssertionError:
                raise ValueError(f"Split {split_id} is not specified in configuration files.")
        else:
            raise TypeError("split must be an int or str.")
        return SplitFilesNamer(parent_dfl=self, split_name=split_str)

    def iter_splits(self):
        for split in self.split_structures:
            yield self.get_split(split)

    @property
    def total_n_ps(self):
        total = 0
        for split in self.iter_splits():
            total += split.n_ps
        return total


class SplitFilesNamer:
    def __init__(self,
                 parent_dfl: DatasetFilesNamer, 
                 split_name: str) -> None:
        logger.debug(f"Running {self.__class__.__name__} in {__name__}")
        self.dfl = parent_dfl
        self.name = split_name

        dir_str_template = self.dfl.dir_to_split_template
        dir_str = dir_str_template.format(split_name=split_name)
        self.path = self.dfl.path / dir_str

        self.cfg_fn = self.dfl.split_cfg_fn

        split_structure = self.dfl.split_structures[split_name]
        self.ps_fidu_fixed: bool = split_structure.ps_fidu_fixed
        self.n_sims: int = split_structure.n_sims

        if self.ps_fidu_fixed:
            self.cmb_ps_fid_path = self.path / self.dfl.cmb_ps_fid_fn
            self.wmap_param_path = self.path / self.dfl.cosmo_param_fn

    @property
    def n_ps(self):
        if self.ps_fidu_fixed:
            return 1
        else:
            return self.n_sims
        
    @property
    def cfg_path(self):
        return self.path / self.cfg_fn

    def get_sim(self, sim_num:int):
        try:
            f"{sim_num:03d}"
        except Exception as e:
            raise ValueError(f"sim_num {sim_num} should be an integer.")
        try:
            assert sim_num < self.n_sims
        except AssertionError:
            raise ValueError(f"Sim {sim_num} is outside the range (max: {self.n_sims}) specified in configuration files.")
        return SimFilesNamer(parent_sfl=self, sim_num=sim_num)

    def iter_sims(self):
        for sim_num in range(self.n_sims):
            yield self.get_sim(sim_num)

    def write_split_conf_file(self, config_yaml) -> None:
        write_conf_file(self.cfg_path, config_yaml)

    def read_split_conf_file(self) -> DictConfig:
        return read_conf_file(self.cfg_path)


def write_conf_file(fp: Path, cfg: DictConfig) -> None:
    # TODO: Handle better?
    try:
        fp.exists()
    except AttributeError as e:
        raise e
    with open(fp, 'w') as f:
        OmegaConf.save(config=cfg, f=f)


def read_conf_file(fp: Path) -> DictConfig:
    # TODO: Handle better?
    try:
        fp.exists()
    except AttributeError as e:
        raise e
    return OmegaConf.load(fp)


class SimFilesNamer:
    def __init__(self,
                 parent_sfl: SplitFilesNamer, 
                 sim_num: int) -> None:
        logger.debug(f"Running {self.__class__.__name__} in {__name__}")
        self.dfl = parent_sfl.dfl
        self.sfl = parent_sfl
        self.sim_num = sim_num

        sim_folder_prefix = self.dfl.sim_folder_prefix
        sim_folder = f"{sim_folder_prefix}{self.sim_num_str}"
        dir_str = self.dfl.dir_to_sim_template.format(sim_folder=sim_folder)
        self.path = self.sfl.path / dir_str

        self.ps_fidu_fixed = self.sfl.ps_fidu_fixed
        self.sim_config_fn = self.dfl.sim_config_fn
        self.cosmo_param_fn = self.dfl.cosmo_param_fn

        self.cmb_map_fid_fn = self.dfl.cmb_map_fid_fn
        self.cmb_ps_fid_fn = self.dfl.cmb_ps_fid_fn
        self.cmb_ps_der_fn = self.dfl.cmb_ps_der_fn
        self.obs_map_fn = self.dfl.obs_map_fn

    @property
    def sim_num_str(self) -> str:
        str_num_digits = self.dfl.sim_str_num_digits
        _sim_num_str = f"{self.sim_num:0{str_num_digits}d}"
        return _sim_num_str

    @property
    def name(self) -> str:
        return f"{self.sfl.name}:{self.sim_num_str}"

    @property
    def sim_config_path(self) -> Path:
        return self.path / self.sim_config_fn

    @property
    def cosmo_param_path(self) -> Path:
        if self.ps_fidu_fixed:
            return self.sfl.path / self.cosmo_param_fn
        else:
            return self.path / self.cosmo_param_fn

    @property
    def cmb_map_fid_path(self) -> Path:
        return self.path / self.cmb_map_fid_fn

    @property
    def cmb_ps_fid_path(self) -> Path:
        if self.ps_fidu_fixed:
            return self.sfl.path / self.cmb_ps_fid_fn
        else:
            return self.path / self.cmb_ps_fid_fn

    @property
    def cmb_ps_der_path(self) -> Path:
        return self.path / self.cmb_ps_der_fn

    def obs_map_path(self, detector:int) -> Path:
        try:
            assert detector in self.sfl.dfl.freqs
        except AssertionError:
            raise ValueError(f"Detector {detector} not found in configuration.")
        return self.path / self.obs_map_fn.format(det=detector)

    def make_folder(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)

    def write_sim_conf_file(self, config: DictConfig) -> None:
        write_conf_file(self.sim_config_path, config)

    def read_sim_conf_file(self) -> DictConfig:
        return read_conf_file(self.sim_config_path)

    def write_wmap_params_file(self, config: DictConfig) -> None:
        write_conf_file(self.cosmo_param_path, config)

    def read_wmap_params_file(self) -> DictConfig:
        return read_conf_file(self.cosmo_param_path)
    
    def write_fid_map(self, cmb_map) -> None:
        map_unit = str(cmb_map.unit)
        hp.write_map(filename=self.cmb_map_fid_path,
                     m=cmb_map,
                     column_units=map_unit,
                     dtype=cmb_map.dtype,
                     overwrite=True)

    def write_obs_map(self, obs_map, freq) -> None:
        if isinstance(obs_map, list):
            obs_map = np.stack(obs_map, axis=0)
        map_unit = str(obs_map.unit)
        hp.write_map(filename=self.obs_map_path(freq),
                     m=obs_map,
                     column_units=map_unit,
                     dtype=obs_map.dtype,
                     overwrite=True)
