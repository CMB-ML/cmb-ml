from pathlib import Path

import numpy as np
import pandas as pd

import camb

from .asset_handlers_base import (
    GenericHandler, 
    register_handler, 
    make_directories)

import logging

logger = logging.getLogger(__name__)


class CambPowerSpectrum(GenericHandler):
    def read(self, path: Path, TT_only=True) -> None:
        """
        Method used to read CAMB's power spectra for analysis.

        Reading CAMB's power spectra for simulation is performed by
           a PySM3 method. We simply provide it with the filepath.
        """
        with open(path, 'r') as file:
            header_line = file.readline().strip().lstrip('#').split()

        # Read the data into a DataFrame, setting the header manually.
        df = pd.read_csv(path, comment='#', sep='\s+', header=None, skiprows=1, names=header_line)
        # L = df['L'].to_numpy()
        TT = df['TT'].to_numpy()
        # EE = df['EE'].to_numpy()
        # BB = df['BB'].to_numpy()
        # TE = df['TE'].to_numpy()
        # PP = df['PP'].to_numpy()
        # PT = df['PT'].to_numpy()
        # PE = df['PE'].to_numpy()
        if TT_only:
            return TT
        else:
            raise NotImplementedError("Untested, no use case currently.")
            return df

    def write(self, path: Path, data: camb.CAMBdata, lmax: int) -> None:
        make_directories(path)
        data.save_cmb_power_spectra(filename=path, lmax=lmax)


class NumpyPowerSpectrum(GenericHandler):
    def read(self, path: Path) -> None:
        return np.load(path)

    def write(self, path: Path, data: np.ndarray) -> None:
        make_directories(path)
        np.save(path, arr=data)


class TextPowerSpectrum(GenericHandler):
    def read(self, path: Path) -> None:
        return np.loadtxt(path)

    def write(self, path: Path, data: np.ndarray) -> None:
        make_directories(path)
        np.savetxt(path, X=data)


register_handler("CambPowerSpectrum", CambPowerSpectrum)
register_handler("NumpyPowerSpectrum", NumpyPowerSpectrum)
register_handler("TextPowerSpectrum", TextPowerSpectrum)
