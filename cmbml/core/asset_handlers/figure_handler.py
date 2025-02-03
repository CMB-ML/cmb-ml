import logging
from pathlib import Path
from typing import Union
# import shutil

from .asset_handlers_base import GenericHandler, make_directories
from .asset_handler_registration import register_handler


logger = logging.getLogger(__name__)


class Figure(GenericHandler):
    def read(self, path: Path) -> None:
        raise NotImplementedError("No read method implemented for Mover Handler; implement a handler for files to be read.")

    def write(self, path: Path) -> Path:
        logger.debug(f"Creating parent directory at {path}")
        make_directories(path)
        return path

register_handler("Figure", Figure)
