from typing import Dict, List, Tuple, Callable
import logging
import re

from omegaconf import DictConfig
from omegaconf import errors as OmegaErrors

from .experiment import ExperimentParameters
from .asset_base_cosmo import Asset
from .namers import Namer
from .split import Split


logger = logging.getLogger(__name__)


class BaseStageExecutor:
    def __init__(self, 
                 cfg: DictConfig, 
                 experiment: ExperimentParameters) -> None:
        # Note that make_asset is a Callable, not a GenericHandler!
        logger.debug("Initializing BaseExecutor")
        self.cfg = cfg
        self.experiment = experiment

        self.name_tracker = Namer(cfg)

        self.stage_str: str  = self.stage_str  # Set in child classes
        self.splits: List[Split] = self.get_applicable_splits()

        self.make_asset: Callable = Asset
        self.assets_in, self.assets_out = self.make_assets()

    def execute(self) -> None:
        # This is the common execution pattern; it may need to be overridden
        logger.debug("Executing BaseExecutor execute() method.")
        for split in self.splits:
            with self.name_tracker.set_context("split", split.name):
                self.process_split(split)

    def process_split(self, split: Split) -> None:
        # Placeholder method to be overridden by subclasses
        logger.debug("Executing BaseExecutor process_split() method.")
        raise NotImplementedError("Subclasses must implement process_split.")

    def get_applicable_splits(self) -> List[Split]:
        all_splits = self.cfg.splits.keys()
        splits_scope = self.cfg.pipeline[self.stage_str].splits
        kinds_of_splits = [kind.lower() for kind in splits_scope]
        patterns = [re.compile(f"^{kind}\\d*$", re.IGNORECASE) for kind in kinds_of_splits]

        filtered_names = []
        for split in all_splits:
            if any(pattern.match(split) for pattern in patterns):
                filtered_names.append(split)
        all_split_objs = [Split(name, self.cfg.splits[name]) for name in filtered_names]
        return all_split_objs

    def make_assets(self) -> Tuple[Dict[str, Asset]]:
        cfg_pipeline = self.cfg.pipeline
        cfg_stage = cfg_pipeline[self.stage_str]
        cfg_assets_out = cfg_stage.assets_out
        assets_out = self.make_assets_out(cfg_assets_out)
        if cfg_stage.get("assets_in", None) is None:
            return None, assets_out
        cfg_assets_in = cfg_stage.assets_in
        assets_in = self.make_assets_in(cfg_assets_in)
        return assets_in, assets_out

    def make_assets_out(self, cfg_assets_out: DictConfig) -> Dict[str, Asset]:
        all_assets_out = {}
        for asset in cfg_assets_out:
            all_assets_out[asset] = self.make_asset(cfg=self.cfg,
                                                    source_stage=self.stage_str,
                                                    asset_name=asset,
                                                    name_tracker=self.name_tracker,
                                                    experiment=self.experiment,
                                                    in_or_out="out")
        return all_assets_out

    def make_assets_in(self, cfg_assets_in: DictConfig) -> Dict[str, Asset]:
        all_assets_in = {}
        for asset in cfg_assets_in:
            try:
                source_pipeline = cfg_assets_in[asset]['stage']
            except OmegaErrors.ConfigAttributeError:
                raise KeyError(f"Error looking up 'stage' for {asset} in {cfg_assets_in}. 'stage' should exist in your pipeline yaml.")
            all_assets_in[asset] = self.make_asset(cfg=self.cfg,
                                                   source_stage=source_pipeline,
                                                   asset_name=asset,
                                                   name_tracker=self.name_tracker,
                                                   experiment=self.experiment,
                                                   in_or_out="in")
        return all_assets_in
