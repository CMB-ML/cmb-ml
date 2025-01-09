from abc import ABC, abstractmethod
# from typing import Dict, List, Tuple, Callable, Union
import logging
# import re
# 
from omegaconf import DictConfig
# from omegaconf import errors as OmegaErrors

# from .asset import Asset, AssetWithPathAlts
from .namers import Namer
from .split import Split
from .config_helper import ConfigHelper
# from .config_helper import get_assets, get_assets_in, get_applicable_splits


logger = logging.getLogger(__name__)


class BaseStageExecutor(ABC):
    def __init__(self, cfg, stage_str):
        self.cfg: DictConfig = cfg
        self.stage_str: str = stage_str
        assert stage_str in cfg.pipeline, f"Stage {stage_str} not found in pipeline yaml."
        self.name_tracker: Namer = Namer(cfg)
        self._config_help = _ch = ConfigHelper(cfg, stage_str)

        self.top_level_working = self._config_help.get_stage_elem_silent("top_level_working")

        self.assets_out = _ch.get_assets_out(name_tracker=self.name_tracker)
        self.assets_in = _ch.get_assets_in(name_tracker=self.name_tracker)

        self.splits = _ch.get_applicable_splits()

    @abstractmethod
    def execute(self):
        raise NotImplementedError("Execute method must be implemented by subclasses.")

    def default_execute(self) -> None:
        # This is the common execution pattern; it may need to be overridden
        assert self.splits is not None, f"Child class, {self.__class__.__name__} has None for splits. Either implement its own execute() or define splits in the pipeline yaml."
        for split in self.splits:
            with self.name_tracker.set_context("split", split.name):
                self.process_split(split)

    def process_split(self, split: Split):
        logger.warning("Executing BaseExecutor process_split() method.")
        raise NotImplementedError("Subclasses must implement process split method.")
    
    @property
    def make_stage_logs(self) -> bool:
        res = self._config_help.get_stage_elem_silent("make_stage_log", self.stage_str)
        if res is None:
            res = False
        return res

    @property
    def model_epochs(self):
        return self._config_help.get_epochs()

    @property
    def map_fields(self):
        return self._config_help.get_map_fields()

    @property
    def override_sim_nums(self):
        """
        Values for this (per the comment) may be a single int, a list of ints, or null.

        Returns either a list of sims, or None
        """
        sim_nums = self._config_help.get_stage_element('override_n_sims', stage_str=self.stage_str)
        try:
            return list(range(sim_nums))
        except TypeError:
            return sim_nums

    def get_stage_element(self, stage_element):
        return self._config_help.get_stage_element(stage_element, stage_str=self.stage_str)

    def ensure_splits(self):
        if len(self.splits) == 0:
            raise ValueError("No splits found in the pipeline configuration for this Executor.")
