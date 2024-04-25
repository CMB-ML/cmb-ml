from pathlib import Path
from typing import Dict

from ...core.asset_handlers import GenericHandler, _make_directories
from ...core.asset_handler_registration import register_handler

from ..physics_cmb import make_cmb_ps

import logging

logger = logging.getLogger(__name__)

class PSHandler(GenericHandler):
    def read(self, path: Path) -> Dict:
        raise NotImplementedError("PowerSpectrum currently writes information only.")

    def write(self, path: Path, cosmo_params, max_ell) -> None:
        path = Path(path)
        _make_directories(path)
        
        make_cmb_ps(cosmo_params, max_ell, path)
        
    

register_handler("PowerSpectrum", PSHandler)