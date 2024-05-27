from typing import Any, Dict, List, Union
from pathlib import Path
import logging

import pandas as pd
import numpy as np

from src.core.asset_handlers import GenericHandler, make_directories
from .asset_handler_registration import register_handler


logger = logging.getLogger(__name__)


class PandasCsvHandler(GenericHandler):
    def read(self, path: Union[Path, str]):
        res = pd.read_csv(path)
        return res

    def write(self, 
              path: Union[Path, str], 
              data: Union[dict, pd.DataFrame],
              ):
        make_directories(path)
        try:
            data.to_csv(path, index=False)
        except:
            pd.DataFrame(data).to_csv(path, index=False)


register_handler("PandasCsvHandler", PandasCsvHandler)
