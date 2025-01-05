from pathlib import Path
import logging

from omegaconf import DictConfig
from omegaconf import errors as OmegaErrors

from .namers import Namer
from .asset_handlers import *
from .asset_handlers.asset_handler_registration import get_handler


logger = logging.getLogger(__name__)


class Asset:
    def __init__(self, cfg, source_stage, asset_name, name_tracker, in_or_out):
        stage_cfg = cfg.pipeline[source_stage]
        asset_info = stage_cfg.assets_out[asset_name]

        self.source_stage_dir = stage_cfg.get('dir_name', None)
        # self.fn = asset_info.get('fn', "")

        self.name_tracker:Namer = name_tracker
        self.can_read = False
        self.can_write = False
        if in_or_out == "in":
            self.can_read = True
        if in_or_out == "out":
            self.can_write = True

        handler: GenericHandler = get_handler(asset_info)
        self.handler = handler()
        # try:
        self.path_template = asset_info.get('path_template', None)
        if self.path_template is None:
            logger.warning("No template found.")
            # TODO: Remove? Think through this?
            raise Exception("No path template found! No known good reasons for this dead end...")

        self.use_fields = asset_info.get("use_fields", None)
        # self.get_other_keys(asset_info)

    # def get_other_keys(self, asset_info):
    #     for k, v in asset_info.items():
    #         if k in ["path_template", "handler"]:
    #             continue
    #         elif k in ["path_template_alt"]:
    #             # Ensure this is not a simple Asset:
    #             assert self.__class__.__name__ != "Asset", "This was created as an 'Asset' instead of an 'AssetWithPathAlts'. Either something has gone wrong in the BaseExecutor or a custom BaseExecutor has been made incorrecty."
    #         elif k in ["fn", "alt_path_template"]:
    #             # No, move this to Check configs if it's a concern. It can happen before runtime.
    #             raise KeyError("The keys 'fn' and 'alt_path_template' are banned (at least during active development).")
    #         else:
    #             self.__setattr__(k, v)

    @property
    def path(self):
        with self.name_tracker.set_context("stage", self.source_stage_dir):
            return self.name_tracker.path(self.path_template)

    def read(self, **kwargs):
        try:
            if self.can_read:
                return self.handler.read(self.path, **kwargs)
        except TypeError as e:
            logger.exception("The calling .read() method must be given keyword arguments only.", exc_info=e)
            raise e

    def write(self, **kwargs):
        # If you're here debugging a "TypeError: write() takes 1 positional argument but 2 were given" error, 
        #   it's because you're calling .write() with positional arguments. 
        # It must be called with keyword arguments only.
        try:
            if self.can_write:
                return self.handler.write(self.path, **kwargs)
        except TypeError as e:
            logger.exception("The calling .write() method must be given keyword arguments only.", exc_info=e)
            raise e

class AssetWithPathAlts(Asset):
    def __init__(self, cfg, source_stage, asset_name, name_tracker, in_or_out):
        super().__init__(cfg, source_stage, asset_name, name_tracker, in_or_out)
        stage_cfg = cfg.pipeline[source_stage]
        asset_info = stage_cfg.assets_out[asset_name]
        self.path_template_alt = asset_info.path_template_alt
    
    @property
    def path_alt(self):
        with self.name_tracker.set_context("stage", self.source_stage_dir):
            return self.name_tracker.path(self.path_template_alt)

    def read(self, use_alt_path:bool=None, **kwargs):
        if use_alt_path is None:
            raise AttributeError("Use alt path must be specified.")
        if self.can_read:
            if use_alt_path:
                return self.handler.read(self.path_alt, **kwargs)
            else:
                return self.handler.read(self.path, **kwargs)

    def write(self, use_alt_path:bool=False, **kwargs):
        if self.can_write:
            if use_alt_path:
                return self.handler.write(self.path_alt, **kwargs)
            else:
                return self.handler.write(self.path, **kwargs)
