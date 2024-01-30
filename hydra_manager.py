from pathlib import Path
from typing import List, Dict, Any
import logging

import numpy as np

from omegaconf import DictConfig, OmegaConf
import hydra


logger = logging.getLogger(__name__)


class Dataset:
    def __init__(self, conf: DictConfig) -> None:
        logger.debug(f"Running {self.__class__.__name__} in {__name__}")
        self.name: str = conf.dataset_name
        self.detectors: List[int] = conf.simulation.detector_freqs
        self.split_structures: Dict[str, Any] = dict(conf.splits)
        self.sim_str_num_digits: int = conf.file_system.sim_str_num_digits
        self.nside = conf.simulation.nside
        self.preset_strings = conf.simulation.preset_strings
        self.freqs = conf.simulation.detector_freqs
        self.components = conf.simulation.component_objects

    # def get_split(self, split:str):
    #     try: 
    #         assert split in self.split_structures.keys()
    #     except AssertionError:
    #         raise ValueError(f"Split {split} is not specified in configuration files.")
    #     return Split(parent=self, split_name=split)

    # def iter_splits(self):
    #     for split in self.split_structures:
    #         yield self.get_split(split)

#     @property
#     def total_n_ps(self):
#         total = 0
#         for split in self.iter_splits():
#             total += split.n_ps
#         return total


# class Split:
#     def __init__(self,
#                  dataset: Dataset, 
#                  split_name: str) -> None:
#         logger.debug(f"Running {self.__class__.__name__} in {__name__}")
#         self.dataset = dataset
#         self.name = split_name

#         split_structure = self.dataset.split_structures[split_name]
#         self.ps_fidu_fixed: bool = split_structure.ps_fidu_fixed
#         self.n_sims: int = split_structure.n_sims

#     @property
#     def n_ps(self):
#         if self.ps_fidu_fixed:
#             return 1
#         else:
#             return self.n_sims
        
#     def get_sim(self, sim_num:int):
#         try:
#             f"{sim_num:03d}"
#         except Exception as e:
#             raise ValueError(f"sim_num {sim_num} should be an integer.")
#         try:
#             assert sim_num < self.n_sims
#         except AssertionError:
#             raise ValueError(f"Sim {sim_num} is outside the range (max: {self.n_sims}) specified in configuration files.")
#         return Sim(split=self, sim_num=sim_num)

#     def iter_sims(self):
#         for sim_num in range(self.n_sims):
#             yield self.get_sim(sim_num)


# class Sim:
#     def __init__(self,
#                  split: Split, 
#                  sim_num: int) -> None:
#         logger.debug(f"Running {self.__class__.__name__} in {__name__}")
#         self.dataset = split.dataset
#         self.split = split
#         self.sim_num = sim_num

#         self.ps_fidu_fixed = self.split.ps_fidu_fixed

#     @property
#     def sim_num_str(self) -> str:
#         str_num_digits = self.dataset.sim_str_num_digits
#         _sim_num_str = f"{self.sim_num:0{str_num_digits}d}"
#         return _sim_num_str

