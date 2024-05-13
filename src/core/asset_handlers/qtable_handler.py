from pathlib import Path
from typing import Dict
from astropy.table import QTable

from .asset_handlers import GenericHandler
from .asset_handler_registration import register_handler

import logging

logger = logging.getLogger(__name__)


class QTableHandler(GenericHandler):
    def read(self, path: Path) -> Dict:
        logger.debug(f"Reading QTable from '{path}'")
        planck_beam_info = QTable.read(path, format="ascii.ipac")
        planck_beam_info.add_index("band")
        return planck_beam_info

    def write(self, path: Path) -> None:
        raise NotImplementedError("QTables currently store information only.")
    

register_handler("QTable", QTableHandler)
