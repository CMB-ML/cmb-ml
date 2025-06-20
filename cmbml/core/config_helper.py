from typing import Any, Dict, List, Optional
from omegaconf import DictConfig, OmegaConf
from omegaconf import errors as OmegaErrors
import re

# import hydra

# import numpy as np

# import torch

from cmbml.core.asset import Asset, AssetWithPathAlts
from cmbml.core.namers import Namer
from cmbml.core.split import Split

# from core.dataset import LabelledCMBMapDataset
from cmbml.utils.planck_instrument import make_instrument, Instrument
# from core.asset_handlers.healpy_map_handler import HealpyMap
# from core.asset_handlers.pytorch_model_handler import PyTorchModel


class ConfigHelper:
    def __init__(self, cfg: DictConfig, stage_str:str=None):
        self.stage_str = stage_str
        self.cfg: DictConfig = cfg

    def get_stage_element(self, 
                          stage_element: str, 
                          stage_str: str=None) -> Any:
        cfg_pipeline = self.cfg.pipeline
        cfg_stage = cfg_pipeline.get(stage_str)
        if cfg_stage is None:
            raise OmegaErrors.ConfigAttributeError(f"The stage '{stage_str}' was not found in the pipeline yamls.")
        try:
            return cfg_stage[stage_element]
        except KeyError:
            raise OmegaErrors.ConfigKeyError(f"The stage '{stage_str}' is missing the key '{stage_element}'")

    def get_stage_elem_silent(self, 
                              stage_element: str, 
                              stage_str: str=None) -> Optional[Dict]:
        if stage_str is None:
            stage_str = self.stage_str
        try:
            return self.get_stage_element(stage_element, stage_str)
        except (OmegaErrors.ConfigKeyError, OmegaErrors.ConfigAttributeError):
            return None

    def get_map_fields(self):
        return self.cfg.scenario.map_fields

    def make_name_tracker(self):
        """
        Returns a Namer. A Namer maintains state, only one should be used.
        """
        return Namer(self.cfg)

    def get_instrument(self):
        return make_instrument(self.cfg)

    def get_split(self, split_name):
        return Split(split_name, self.cfg.splits[split_name])

    def get_assets_out(self, name_tracker:Namer, stage_str: str=None) -> Dict[str, Asset]:
        # assets_out = self.get_stage_elem_silent("assets_out", stage_str=stage_str)
        if stage_str is None:
            stage_str = self.stage_str
        return get_assets(self.cfg, stage_str, name_tracker, in_or_out="out")

    def get_assets_in(self, name_tracker:Namer, stage_str: str=None) -> Dict[str, Asset]:
        if stage_str is None:
            stage_str = self.stage_str
        return get_assets_in(self.cfg, stage_str, name_tracker)

    def get_applicable_splits(self, stage_str: str=None) -> List[Split]:
        if stage_str is None:
            stage_str = self.stage_str
        return get_applicable_splits(self.cfg, stage_str)

    def get_some_asset_out(self, name_tracker, asset, stage_str):
        assets = self.get_assets_out(name_tracker, stage_str)
        return assets[asset]

    def get_some_asset_in(self, name_tracker, asset, stage_str):
        assets = self.get_assets_in(name_tracker, stage_str)
        return assets[asset]

    def get_epochs(self, stage_str=None):
        epochs = self.get_stage_elem_silent(stage_element="epochs")
        return epochs

    def get_override_sim_ns(self, stage_str=None):
        or_sn = self.get_stage_elem_silent(stage_element="override_n_sims")
        return or_sn


def create_asset_instance(asset_type: str, cfg: DictConfig, source_stage: str, asset_name: str, name_tracker: Namer, in_or_out: str):
    AssetClass = AssetWithPathAlts if asset_type == 'path_template_alt' else Asset
    return AssetClass(cfg=cfg, source_stage=source_stage, asset_name=asset_name, name_tracker=name_tracker, in_or_out=in_or_out)


def get_assets(cfg: DictConfig, stage_str: str, name_tracker: Namer, in_or_out: str) -> Dict[str, Asset]:
    config_handler = ConfigHelper(cfg)
    # In the pipeline yaml,  Asset Information is where an asset is created; we need "asset_out" in the next line
    assets_info = config_handler.get_stage_elem_silent("assets_out", stage_str)
    assets = {}
    if assets_info:
        for asset_name, asset_details in assets_info.items():
            asset_type = 'path_template_alt' if 'path_template_alt' in asset_details else 'normal'
            assets[asset_name] = create_asset_instance(asset_type, cfg, stage_str, asset_name, name_tracker, in_or_out)
    return assets


def get_assets_in(cfg: DictConfig, stage_str: str, name_tracker: Namer) -> Dict[str, Asset]:
    """
    Prepares input assets by fetching details from the stage where each asset is defined as output.
    """
    config_handler = ConfigHelper(cfg)
    assets_in_info = config_handler.get_stage_elem_silent("assets_in", stage_str)
    assets_in = {}
    if assets_in_info:
        for asset_name, details in assets_in_info.items():
            details = dict(details)
            source_stage = details.pop('stage')

            # Allow for the case where the asset name is not the same as the original name
            orig_name = details.pop('orig_name', asset_name)

            assets_out_at_source_info = config_handler.get_stage_element("assets_out", source_stage)
            if orig_name in assets_out_at_source_info:
                assets_in[asset_name] = get_assets(cfg, source_stage, name_tracker, "in")[orig_name]
            else:
                raise ValueError(f"Asset '{orig_name}' not found in stage '{source_stage}' outputs.")

            assets_in[asset_name].path_overrides = details
    return assets_in


def get_applicable_splits(cfg: DictConfig, stage_str: str) -> List[Split]:
    config_helper = ConfigHelper(cfg)
    # All splits are defined in the splits yaml
    splits_all_cfg = cfg.splits
    splits_all     = [k for k in splits_all_cfg.keys() if k != 'name']
    
    # In a stage in a pipeline yaml, we may specify which splits to work on
    #    This is generic, e.g. for all {Test0, Test1}, we may list just "test"
    splits_scope = config_helper.get_stage_elem_silent("splits", stage_str)

    # If no splits are listed for a stage, bail out
    if splits_scope is None:
        return []

    # Use regex to search for all matching Test# in all_splits
    patterns = [re.compile(f"^{kind}(?:[A-Z])?(?:\\d{{1,4}})?$", re.IGNORECASE) for kind in splits_scope]
    filtered_names = [name for name in splits_all if any(pattern.match(name) for pattern in patterns)]

    # Create a Split object for each of the splits we found
    applicable_splits = [Split(name, splits_all_cfg[name]) for name in filtered_names]
    return applicable_splits
