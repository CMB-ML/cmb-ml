from pathlib import Path
from typing import Dict
from astropy.table import QTable

from .asset_handlers_base import GenericHandler
from .asset_handler_registration import register_handler

import logging

logger = logging.getLogger(__name__)


class QTableHandler(GenericHandler):
    def read(self, path: Path) -> Dict:
        logger.debug(f"Reading QTable from '{path}'")
        try:
            planck_beam_info = QTable.read(path, format="ascii.ipac")
        except TypeError as e:
            # When the file is not found, astropy raises a TypeError. This may also happen if the file is not in the correct format.
            raise IOError(f"Error reading QTable from '{path}': {e}. Have you gotten the Science Assets?")
        planck_beam_info.add_index("band")
        return planck_beam_info

    def write(self, path: Path) -> None:
        raise NotImplementedError("QTables currently store information only.")
    

register_handler("QTable", QTableHandler)
