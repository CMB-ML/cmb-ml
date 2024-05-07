from typing import Dict, List, Tuple, Callable, Union
import logging
import re

from omegaconf import DictConfig, OmegaConf
from omegaconf import errors as OmegaErrors

from .experiment import ExperimentParameters
from .asset import Asset
from .namers import Namer
from .split import Split


logger = logging.getLogger(__name__)


class BaseStageExecutor:
    def __init__(self, 
                 cfg: DictConfig, 
                 experiment: ExperimentParameters,
                 stage_str: str) -> None:
        self.cfg = cfg
        self.experiment = experiment

        self.name_tracker = Namer(cfg)

        self.stage_str: str  = stage_str
        self._ensure_stage_string_in_pipeline_yaml()

        self.splits: Union[List[Split], None] = self._get_applicable_splits()
        self.assets_out: Union[Dict[str, Asset], None] = self._make_assets_out()
        self.assets_in:  Union[Dict[str, Asset], None] = self._make_assets_in()

    def _ensure_stage_string_in_pipeline_yaml(self):
        # We do not know the contents of python code (the stage executors) until runtime.
        #    TODO: Checking this early would require changes to the PipelineContext
        #    and possibly include making stage_str into a class variable. Probably a good idea, in hindsight.
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
        logger.warning("Executing BaseExecutor process_split() method.")
        raise NotImplementedError("Subclasses must implement process_split if it is to be used.")

    def _get_stage_element(self, stage_element="assets_out"):
        """
        Supported stage elements are "assets_in", "assets_out", and "splits"
        raises omegaconf.errors.ConfigAttributeError if the stage in the pipeline yaml is empty (e.g. the CheckHydraConfigs stage).
        raises omegaconf.errors.ConfigKeyError if the stage in the pipeline yaml is missing assets_in or assets_out.
        """
        cfg_pipeline = self.cfg.pipeline
        cfg_stage = cfg_pipeline[self.stage_str]
        if cfg_stage is None:
            raise OmegaErrors.ConfigAttributeError
        stage_element = cfg_stage[stage_element]  # OmegaErrors.ConfigKeyError from here
        return stage_element

    def _get_applicable_splits(self) -> List[Split]:
        # Pull specific splits for this stage from the pipeline hydra config
        try:
            splits_scope = self._get_stage_element(stage_element='splits')
        except (OmegaErrors.ConfigKeyError, OmegaErrors.ConfigAttributeError):
            # Or None if the pipeline has no "splits" for this stage
            return None
        # Get all possible splits from the splits hydra config
        all_splits = self.cfg.splits.keys()

        # Make a regex pattern to find "test" in "Test6"
        kinds_of_splits = [kind.lower() for kind in splits_scope]
        patterns = [re.compile(f"^{kind}\\d*$", re.IGNORECASE) for kind in kinds_of_splits]

        filtered_names = []
        for split in all_splits:
            if any(pattern.match(split) for pattern in patterns):
                filtered_names.append(split)
        # Create a Split for all splits to which we want to apply this pipeline stage
        all_split_objs = [Split(name, self.cfg.splits[name]) for name in filtered_names]
        return all_split_objs

    def _make_assets_out(self) -> Dict[str, Asset]:
        # Pull the list of output assets for this stage from the pipeline hydra config
        try:
            cfg_assets_out = self._get_stage_element(stage_element="assets_out")
        except (OmegaErrors.ConfigKeyError, OmegaErrors.ConfigAttributeError):
            # Or None if the pipeline has no "assets_out" for this stage
            return None
        
        # Create assets directly
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
        # Pull the list of input assets for this stage from the pipeline hydra config
        try:
            cfg_assets_in = self._get_stage_element(stage_element="assets_in")
        except (OmegaErrors.ConfigKeyError, OmegaErrors.ConfigAttributeError):
            # Or None if the pipeline has no "assets_out" for this stage
            return None
        all_assets_in = {}
        # Create assets by looking up the stage in which the asset was originally created
        for asset in cfg_assets_in:
            source_pipeline = cfg_assets_in[asset]['stage']
            all_assets_in[asset] = Asset(cfg=self.cfg,
                                          source_stage=source_pipeline,
                                          asset_name=asset,
                                          name_tracker=self.name_tracker,
                                          experiment=self.experiment,
                                          in_or_out="in")
        return all_assets_in