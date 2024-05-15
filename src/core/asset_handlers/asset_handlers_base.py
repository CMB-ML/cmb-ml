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


class Config(GenericHandler):
    def read(self, path: Path) -> Dict:
        logger.debug(f"Reading config from '{path}'")
        with open(path, 'r') as infile:
            data = yaml.safe_load(infile)
        return data

    def write(self, path, data) -> None:
        logger.debug(f"Writing config to '{path}'")
        make_directories(path)
        unnumpy_data = _convert_numpy(data)
        with open(path, 'w') as outfile:
            yaml.dump(unnumpy_data, outfile, default_flow_style=False)


class Mover(GenericHandler):
    def read(self, path: Path) -> None:
        raise NotImplementedError("No read method implemented for Mover Handler; implement a handler for files to be read.")

    def write(self, path: Path, source_location: Union[Path, str]) -> None:
        make_directories(path)
        # Move the file from the temporary location
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


def _convert_numpy(obj: Union[Dict[str, Any], List[Any], np.generic]) -> Any:
    # Recursive function
    # The `Any`s in the above signature should be the same as the signature itself
    # GPT4, minimal modification
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: _convert_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy(item) for item in obj]
    else:
        return obj


def make_directories(path: Union[Path, str]) -> None:
    path = Path(path)
    folders = path.parent
    folders.mkdir(exist_ok=True, parents=True)


register_handler("EmptyHandler", EmptyHandler)
register_handler("Config", Config)
register_handler("Mover", Mover)
