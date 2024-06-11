from typing import Dict
from pathlib import Path
import logging

from astropy.table import QTable

from cmbml.core import GenericHandler
from cmbml.core import register_handler


logger = logging.getLogger(__name__)


class QTableHandler(GenericHandler):
    def read(self, path: Path) -> Dict:
        logger.debug(f"Reading QTable from '{path}'")
        planck_beam_info = QTable.read(path, format="ascii.ipac")
        planck_beam_info.add_index("band")
        return planck_beam_info

    def write(self, path, data) -> None:
        raise NotImplementedError("This asset can only be read.")


register_handler("QTable", QTableHandler)
