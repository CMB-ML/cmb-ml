from typing import Dict, List, Tuple, Callable, Union
import logging
import re

from omegaconf import DictConfig
from omegaconf import errors as OmegaErrors

from .experiment import ExperimentParameters
from .asset import Asset
from .namers import Namer
from .split import Split


logger = logging.getLogger(__name__)


class BaseStageExecutor:
    def __init__(self, 
                 cfg: DictConfig, 
                 experiment: ExperimentParameters) -> None:
        self.cfg = cfg
        self.experiment = experiment

        self.name_tracker = Namer(cfg)

        self.stage_str: str  = self.stage_str  # Set in child classes
        self._ensure_stage_string_in_pipeline_yaml()

        self.splits: Union[List[Split], None] = self._get_applicable_splits()
        self.assets_out: Union[Dict[str, Asset], None] = self._make_assets_out()
        self.assets_in:  Union[Dict[str, Asset], None] = self._make_assets_in()

    def _ensure_stage_string_in_pipeline_yaml(self):
        assert self.stage_str in self.cfg.pipeline, f"Stage string for child class {self.__class__.__name__} not found." + \
             " Ensure that this particular Executor has set a stage_str matching a stage in the pipeline yaml."

    def execute(self) -> None:
        # This is the common execution pattern; it may need to be overridden
        logger.debug("Executing BaseExecutor execute() method.")
        assert self.splits is not None, f"Child class, {self.__class__.__name__} has None for splits. Either implement its own execute() or define splits in the pipeline yaml."
        for split in self.splits:
            with self.name_tracker.set_context("split", split.name):
                self.process_split(split)

    def process_split(self, split: Split) -> None:
        # Placeholder method to be overridden by subclasses
        logger.debug("Executing BaseExecutor process_split() method.")
        raise NotImplementedError("Subclasses must implement process_split.")

    def _get_applicable_splits(self) -> List[Split]:
        cfg_pipeline = self.cfg.pipeline
        try:
            splits_scope = cfg_pipeline['splits']
        except OmegaErrors.ConfigKeyError:
            return None
        all_splits = self.cfg.splits.keys()
        kinds_of_splits = [kind.lower() for kind in splits_scope]
        patterns = [re.compile(f"^{kind}\\d*$", re.IGNORECASE) for kind in kinds_of_splits]

        filtered_names = []
        for split in all_splits:
            if any(pattern.match(split) for pattern in patterns):
                filtered_names.append(split)
        all_split_objs = [Split(name, self.cfg.splits[name]) for name in filtered_names]
        return all_split_objs

    def _get_assets_list(self, which_assets="assets_out"):
        """
        raises omegaconf.errors.ConfigAttributeError if the stage in the pipeline yaml is empty (e.g. the CheckHydraConfigs stage).
        raises omegaconf.errors.ConfigKeyError if the stage in the pipeline yaml is missing assets_in or assets_out.
        """
        cfg_pipeline = self.cfg.pipeline
        cfg_stage = cfg_pipeline[self.stage_str]
        if cfg_stage is None:
            raise OmegaErrors.ConfigAttributeError
        cfg_assets = cfg_stage[which_assets]  # OmegaErrors.ConfigKeyError from here
        return cfg_assets

    def _make_assets_out(self) -> Dict[str, Asset]:
        try:
            cfg_assets_out = self._get_assets_list(which_assets="assets_out")
        except (OmegaErrors.ConfigKeyError, OmegaErrors.ConfigAttributeError):
            return None
        all_assets_out = {}
        for asset in cfg_assets_out:
            all_assets_out[asset] = Asset(cfg=self.cfg,
                                          source_stage=self.stage_str,
                                          asset_name=asset,
                                          name_tracker=self.name_tracker,
                                          experiment=self.experiment,
                                          in_or_out="out")
        return all_assets_out

    def _make_assets_in(self) -> Dict[str, Asset]:
        try:
            cfg_assets_in = self._get_assets_list(which_assets="assets_in")
        except (OmegaErrors.ConfigKeyError, OmegaErrors.ConfigAttributeError):
            return None
        all_assets_in = {}
        for asset in cfg_assets_in:
            try:
                source_pipeline = cfg_assets_in[asset]['stage']
            except OmegaErrors.ConfigAttributeError:
                raise KeyError(f"Error looking up 'stage' for {asset} in {cfg_assets_in}. 'stage' should exist in your pipeline yaml.")
            all_assets_in[asset] = Asset(cfg=self.cfg,
                                          source_stage=source_pipeline,
                                          asset_name=asset,
                                          name_tracker=self.name_tracker,
                                          experiment=self.experiment,
                                          in_or_out="in")
        return all_assets_in
