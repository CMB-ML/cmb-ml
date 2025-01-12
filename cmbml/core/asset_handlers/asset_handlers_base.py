from typing import Any, Dict, List, Union
import shutil
from pathlib import Path
import yaml
import logging

import numpy as np

from .asset_handler_registration import register_handler

logger = logging.getLogger(__name__)


class GenericHandler:
    def read(self, path: Path):
        raise NotImplementedError("This read() should be implemented by children classes.")

    def write(self, path: Path, data: Any):
        raise NotImplementedError("This write() should be implemented by children classes.")


class EmptyHandler(GenericHandler):
    def read(self, path: Path):
        raise NotImplementedError("This is a no-operation placeholder and has no read() function.")

    def write(self, path: Path, data: Any=None):
        if data:
            raise NotImplementedError("This is a no-operation placeholder and has no write() function.")
        make_directories(path)


class PlainText(GenericHandler):
    # TODO Find uses of this class and compare with TextHandler
    def read(self, path: Path, astype=str) -> str:
        # logger.debug(f"Reading config from '{path}'")
        try:
            with open(path, 'r', encoding='utf-8') as infile:
                data = infile.read()
        except Exception as e:
            logger.error(f"Failed to read file at '{path}': {e}")
            raise

        if len(data) == 0:
            logger.warning(f"Empty file at '{path}'")

        try:
            data = astype(data)
        except Exception as e:
            if len(data) > 17:
                data = data[:17] + "..."
            logger.error(f"Failed to convert data ({data}) to {astype}: {e}")
            raise

        return data

    def write(self, path, data) -> None:
        logger.debug(f"Writing config to '{path}'")
        data = str(data)
        make_directories(path)
        try:
            with open(path, 'w', encoding='utf-8') as outfile:
                outfile.write(data)
        except Exception as e:
            logger.error(f"Failed to read file at '{path}': {e}")
            raise

class Mover(GenericHandler):
    def read(self, path: Path) -> None:
        raise NotImplementedError("No read method implemented for Mover Handler; implement a handler for files to be read.")

    def write(self, path: Path, source_location: Union[Path, str]) -> None:
        make_directories(path)
        # Move the file from the temporary location (cwd)
        destination_path = Path(path).parent / str(source_location)
        logger.debug(f"Moving from {source_location} to {destination_path}")
        try:
            # Duck typing for more meaningful error messages
            source_path = Path(source_location)
        except Exception as e:
            # TODO: Better except here
            raise e
        shutil.copy(source_path, destination_path)
        source_path.unlink()


def make_directories(path: Union[Path, str]) -> None:
    path = Path(path)
    folders = path.parent
    folders.mkdir(exist_ok=True, parents=True)


register_handler("EmptyHandler", EmptyHandler)
register_handler("Mover", Mover)
register_handler("PlainText", PlainText)
