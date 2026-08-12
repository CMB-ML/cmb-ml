import logging
from typing import Union, Dict
from pathlib import Path

import numpy as np

from .asset_handlers_base import (
    GenericHandler, 
    register_handler, 
    make_directories)


logger = logging.getLogger(__name__)


class NpyHandler(GenericHandler):
    def read(self, path: Path) -> None:
        return np.load(path)

    def write(self, path: Path, data: np.ndarray) -> None:
        make_directories(path)
        np.save(path, arr=data)


class NpzHandler(GenericHandler):
    def read(self, path: Union[Path, str]) -> Dict[str, np.ndarray]:
        with np.load(Path(path)) as f:
            return {k: f[k] for k in f.files}

    def write(self, path: Union[Path, str], data: Dict[str, np.ndarray]) -> None:
        path = Path(path)
        make_directories(path)
        np.savez(path, **data)


register_handler("NpyHandler", NpyHandler)
register_handler("NpzHandler", NpzHandler)
