from typing import Any, Dict, List, Union
from pathlib import Path
import logging

import numpy as np
import healpy as hp
import astropy.units as u
import astropy.io.fits as fits

from .asset_handlers_base import (
    GenericHandler, 
    register_handler, 
    make_directories)
from cmbml.utils.physics_units import get_fields_units_from_fits


logger = logging.getLogger(__name__)


LFI = [30,44,70]


def get_hdu(nom_freq):
    return f"BANDPASS_{'F' if nom_freq not in LFI else ''}{nom_freq:03d}"


class RIMO(GenericHandler):
    """
    Currently, this returns just the wavenumber (GHz) and transmission
    """
    def read(self,
             path: Union[Path, str],
             nom_freq: int
             ):

        # Ensure path is Path, and path exists
        path = Path(path)
        if not Path(path).exists():
            raise FileNotFoundError(f'No such file as "{path}"')

        # Open RIMO
        hdul = fits.open(path)

        # Get label for hdu
        hdu_label = get_hdu(nom_freq)
        hdu = hdul[hdu_label]

        # Get unit for wavenumber (either GHz or 1/cm)
        freq_unit = u.Unit(hdu.columns['WAVENUMBER'].unit)

        # Get freqs
        freq = hdu.data['WAVENUMBER'] * freq_unit
        freq = freq.to(u.GHz, equivalencies=u.spectral())

        # Get transmission
        tx = hdu.data['TRANSMISSION'].astype(float)

        return freq, tx

    def write(self, path: Path) -> None:
        raise NotImplementedError("RIMO stores information only.")


register_handler("RIMO", RIMO)
