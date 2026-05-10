from pathlib import Path
import logging
from typing import Optional

from omegaconf import errors as OmegaErrors

from .namers import Namer
from .asset_handlers.asset_handlers_base import GenericHandler
from .asset_handlers.asset_handler_registration import get_handler
from .failed_path_interp import FailedPathInterpolationSentinel


logger = logging.getLogger(__name__)


class Asset:
    supports_alt_path = False

    def __init__(self, cfg, source_stage, asset_name, name_tracker, in_or_out):
        stage_cfg = cfg.pipeline[source_stage]
        asset_info = stage_cfg.assets_out[asset_name]

        self.source_stage_dir = stage_cfg.get("dir_name", None)
        self.name_tracker: Namer = name_tracker

        self.can_read = in_or_out == "in"
        self.can_write = in_or_out == "out"

        handler: GenericHandler = get_handler(asset_info, source_stage)
        self.handler = handler()

        self.path_template = self._get_path_template(
            asset_info=asset_info,
            asset_name=asset_name,
            source_stage=source_stage,
        )

        self.use_fields = asset_info.get("use_fields", None)
        self.file_size = asset_info.get("file_size", None)
        self.path_overrides = {}

    def _get_path_template(self, *, asset_info, asset_name, source_stage):
        try:
            path_template = asset_info.get("path_template", None)
        except OmegaErrors.InterpolationKeyError as e:
            return FailedPathInterpolationSentinel(
                asset_name=asset_name,
                source_stage=source_stage,
                error=e,
            )

        if path_template is None:
            raise ValueError(
                f"No path_template found for asset {asset_name!r} "
                f"from source stage {source_stage!r}."
            )

        return path_template

    def _path_from_template(self, path_template):
        with self.name_tracker.set_context("stage", self.source_stage_dir):
            if self.path_overrides:
                with self.name_tracker.set_contexts(self.path_overrides):
                    return self.name_tracker.path(path_template)
            return self.name_tracker.path(path_template)

    @property
    def path(self):
        return self._path_from_template(self.path_template)

    def resolve_path(self, *, use_alt_path: Optional[bool] = None, for_write: bool = False):
        if use_alt_path:
            raise ValueError(f"{type(self).__name__} does not support alternate paths.")

        return self.path

    def read(self, *args, use_alt_path: Optional[bool] = None, **kwargs):
        if args:
            raise TypeError(
                f"{type(self).__name__}.read() only accepts keyword arguments. "
                f"Received positional args: {args!r}"
            )

        if not self.can_read:
            raise AttributeError("This asset is not set up to read.")

        path = self.resolve_path(use_alt_path=use_alt_path, for_write=False)
        return self.handler.read(path, **kwargs)

    def start(self, *args, use_alt_path: Optional[bool] = None, **kwargs):
        if args:
            raise TypeError(
                f"{type(self).__name__}.start() only accepts keyword arguments. "
                f"Received positional args: {args!r}"
            )

        if not self.can_write:
            raise AttributeError("This asset is not set up to write.")

        path = self.resolve_path(use_alt_path=use_alt_path, for_write=True)
        return self.handler.start(path, **kwargs)

    def write(self, *args, use_alt_path: Optional[bool] = None, **kwargs):
        if args:
            raise TypeError(
                f"{type(self).__name__}.write() only accepts keyword arguments. "
                f"Received positional args: {args!r}"
            )

        if not self.can_write:
            raise AttributeError("This asset is not set up to write.")

        path = self.resolve_path(use_alt_path=use_alt_path, for_write=True)
        return self.handler.write(path, **kwargs)

    def append(self, *args, **kwargs):
        if not self.can_write:
            raise AttributeError("This asset is not set up to write.")

        if self.handler.append is None:
            raise AttributeError("The handler for this asset does not have an append method.")

        return self.handler.append(*args, **kwargs)


class AssetWithPathAlts(Asset):
    supports_alt_path = True

    def __init__(self, cfg, source_stage, asset_name, name_tracker, in_or_out):
        super().__init__(cfg, source_stage, asset_name, name_tracker, in_or_out)

        stage_cfg = cfg.pipeline[source_stage]
        asset_info = stage_cfg.assets_out[asset_name]

        self.path_template_alt = asset_info.path_template_alt

    @property
    def path_alt(self):
        return self._path_from_template(self.path_template_alt)

    def resolve_path(self, *, use_alt_path: Optional[bool] = None, for_write: bool = False):
        if use_alt_path is None:
            if for_write:
                raise ValueError(
                    f"{type(self).__name__}.write()/start() must specify "
                    "use_alt_path=True or use_alt_path=False."
                )

            # For reads, default to the alternate/shared path.
            use_alt_path = True

        return self.path_alt if use_alt_path else self.path