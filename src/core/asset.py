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
            raise Exception("No path template found! Jim is checking for good reasons for this dead end...")
        # except OmegaErrors.ConfigAttributeError:
        #     self.path_template = cfg.file_system.default_path_template

    @property
    def path(self):
        with self.name_tracker.set_context("stage", self.source_stage_dir):
            # with self.name_tracker.set_context("fn", self.fn):
            return self.name_tracker.path(self.path_template)

    def read(self, **kwargs):
        if self.can_read:
            return self.handler.read(self.path, **kwargs)

    def write(self, **kwargs):
        if self.can_write:
            return self.handler.write(self.path, **kwargs)


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

    def read(self, use_alt_path:bool=False, **kwargs):
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
