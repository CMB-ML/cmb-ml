# from abc import ABC, abstractmethod
# from typing import Dict, List, Tuple, Callable, Union
# import logging
# import re

# from omegaconf import DictConfig, OmegaConf
# from omegaconf import errors as OmegaErrors

# from .asset import Asset, AssetWithPathAlts
# from .namers import Namer
# from .split import Split


# logger = logging.getLogger(__name__)


# class BaseStageExecutor:
#     def __init__(self, 
#                  cfg: DictConfig, 
#                 #  experiment: ExperimentParameters,
#                  stage_str: str) -> None:
#         self.cfg = cfg
#         # self.experiment = experiment

#         self.name_tracker = Namer(cfg)

#         self.stage_str: str = stage_str
#         self._ensure_stage_string_in_pipeline_yaml()

#         self.splits: Union[List[Split], None] = self._get_applicable_splits()
#         self.assets_out: Union[Dict[str, Asset], None] = self._make_assets_out()
#         self.assets_in:  Union[Dict[str, Asset], None] = self._make_assets_in()

#         self.model_epochs: Union[List[Union[str, int]], None] = self._get_epochs()

#     def _ensure_stage_string_in_pipeline_yaml(self):
#         # We do not know the contents of python code (the stage executors) until runtime.
#         #    TODO: Checking this early would require changes to the PipelineContext
#         #    and possibly include making stage_str into a class variable. Probably a good idea, in hindsight.
#         assert self.stage_str in self.cfg.pipeline, f"Stage string for child class {self.__class__.__name__} not found." + \
#              " Ensure that this particular Executor has set a stage_str matching a stage in the pipeline yaml."

#     @abstractmethod
#     def execute(self) -> None:
#         """
#         All executors should implement an execute method for clarity.
#         The default_execute() method below may suffice for many.
#         """
#         raise NotImplementedError("The is an abstract method.")

#     def default_execute(self) -> None:
#         # This is the common execution pattern; it may need to be overridden
#         logger.debug("Executing BaseExecutor execute() method.")
#         assert self.splits is not None, f"Child class, {self.__class__.__name__} has None for splits. Either implement its own execute() or define splits in the pipeline yaml."
#         for split in self.splits:
#             with self.name_tracker.set_context("split", split.name):
#                 self.process_split(split)

#     def process_split(self, split: Split) -> None:
#         # Placeholder method to be overridden by subclasses
#         logger.warning("Executing BaseExecutor process_split() method.")
#         raise NotImplementedError("Subclasses must implement process_split if it is to be used.")

#     @property
#     def make_stage_logs(self) -> bool:
#         res = self.get_stage_elem_silent("make_stage_log", self.stage_str)
#         if res is None:
#             res = False
#         return res

#     def _get_applicable_splits(self) -> List[Split]:
#         # Pull specific splits for this stage from the pipeline hydra config
#         splits_scope = self.get_stage_elem_silent(stage_element='splits', 
#                                                    stage_str=self.stage_str)

#         if splits_scope is None:
#             return None

#         # Get all possible splits from the splits hydra config
#         all_splits = [key for key in self.cfg.splits.keys() if key != 'name']

#         # Make a regex pattern to find, e.g., "test" in "Test6"
#         kinds_of_splits = [kind.lower() for kind in splits_scope]
#         patterns = [re.compile(f"^{kind}\\d*$", re.IGNORECASE) for kind in kinds_of_splits]

#         filtered_names = []
#         for split in all_splits:
#             if any(pattern.match(split) for pattern in patterns):
#                 filtered_names.append(split)
#         # Create a Split for all splits to which we want to apply this pipeline stage
#         all_split_objs = [Split(name, self.cfg.splits[name]) for name in filtered_names]
#         return all_split_objs

#     def _make_assets_out(self) -> Dict[str, Asset]:
#         # Pull the list of output assets for this stage from the pipeline hydra config
#         cfg_assets_out = self.get_stage_elem_silent(stage_element="assets_out",
#                                                     stage_str=self.stage_str)
#         if cfg_assets_out is None:
#             return None

#         # Create assets directly
#         all_assets_out = {}
#         for asset in cfg_assets_out:
#             use_asset = self._get_asset_type(cfg_assets_out[asset])
#             all_assets_out[asset] = use_asset(cfg=self.cfg,
#                                               source_stage=self.stage_str,
#                                               asset_name=asset,
#                                               name_tracker=self.name_tracker,
#                                               in_or_out="out")
#         return all_assets_out

#     def _make_assets_in(self) -> Dict[str, Asset]:
#         # Pull the list of input assets for this stage from the pipeline hydra config
#         cfg_assets_in = self.get_stage_elem_silent(stage_element="assets_in",
#                                                     stage_str=self.stage_str)
#         if cfg_assets_in is None:
#             return None

#         all_assets_in = {}
#         # Create assets by looking up the stage in which the asset was originally created
#         for asset in cfg_assets_in:
#             source_stage = cfg_assets_in[asset]['stage']

#             cfg_assets_out = self.get_stage_element(stage_element="assets_out", 
#                                                      stage_str=source_stage)
            
#             # If an original name is specified, grab that asset. Otherwise, get this one.
#             orig_name = cfg_assets_in[asset].get('orig_name', asset)
#             use_asset = self._get_asset_type(cfg_assets_out[orig_name])
#             all_assets_in[asset] = use_asset(cfg=self.cfg,
#                                              source_stage=source_stage,
#                                              asset_name=orig_name,
#                                              name_tracker=self.name_tracker,
#                                              in_or_out="in")
#         return all_assets_in

#     @staticmethod
#     def _get_asset_type(asset):
#         if 'path_template_alt' in asset:
#             use_asset = AssetWithPathAlts
#         else:
#             use_asset = Asset
#         return use_asset

#     def _get_epochs(self):
#         epochs = self.get_stage_elem_silent(stage_element="epochs")
#         return epochs

#     def get_override_sim_nums(self):
#         """
#         Values for this (per the comment) may be a single int, a list of ints, or null.

#         Returns either a list of sims, or None
#         """
#         sim_nums = self.get_stage_element('override_n_sims')
#         try:
#             return list(range(sim_nums))
#         except TypeError:
#             return sim_nums

#     def get_stage_element(self, stage_element="assets_out", stage_str=None):
#         """
#         Supported stage elements are "assets_in", "assets_out", and "splits"
#         raises omegaconf.errors.ConfigAttributeError if the stage in the pipeline yaml is empty (e.g. the CheckHydraConfigs stage).
#         raises omegaconf.errors.ConfigKeyError if the stage in the pipeline yaml is missing assets_in or assets_out.
#         """
#         cfg_pipeline = self.cfg.pipeline
#         if stage_str is None:
#             stage_str = self.stage_str
#         cfg_stage = cfg_pipeline[stage_str]
#         if cfg_stage is None:
#             raise OmegaErrors.ConfigAttributeError(f"The stage '{cfg_stage}' was not found in the pipeline yamls.")
#         try:
#             stage_element = cfg_stage[stage_element]  # OmegaErrors.ConfigKeyError from here
#         except OmegaErrors.ConfigKeyError:
#             raise OmegaErrors.ConfigKeyError(f"The stage '{cfg_stage}' is missing the key '{stage_element}'")
#         return stage_element

#     def get_stage_elem_silent(self, stage_element="assets_out", stage_str=None):
#         """
#         Swallows errors. Use when applying to optional stage elements in the pipeline.
#         """
#         try:
#             res = self.get_stage_element(stage_element, stage_str)
#         except (OmegaErrors.ConfigKeyError, OmegaErrors.ConfigAttributeError):
#             # Or None if the pipeline has no "splits" for this stage
#             return None
#         return res
