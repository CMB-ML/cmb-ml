from typing import Any, Dict, List, Union
from pathlib import Path
import logging

import pandas as pd
import numpy as np

from cmbml.core.asset_handlers import GenericHandler, make_directories
from .asset_handler_registration import register_handler


logger = logging.getLogger(__name__)


class TextHandler(GenericHandler):
    # TODO Find uses of this class and compare with PlainText
    def read(self, path: Union[Path, str]):
        path = Path(path)
        with open(path, 'r') as f:
            res = f.read()
        return res

    def write(self, 
              path: Union[Path, str], 
              data: str
              ):
        path = Path(path)
        make_directories(path)
        with open(path, 'w') as f:
            f.write(data)


register_handler("TextHandler", TextHandler)
