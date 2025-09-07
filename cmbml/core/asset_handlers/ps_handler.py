from pathlib import Path

import numpy as np
import pandas as pd

import camb

from .asset_handlers_base import (
    GenericHandler, 
    register_handler, 
    make_directories)
from cmbml.utils.planck_ps_text_add_l01 import add_missing_multipoles

import logging


logger = logging.getLogger(__name__)


class CambPowerSpectrum(GenericHandler):
    """
    Reads power spectrum files using Pandas, writes using CAMB

    Power spectra read are returned with only TT information.
    """
    def read(self, path: Path, TT_only=True) -> None:
        """
        Method used to read CAMB's power spectra for analysis.

        Reading CAMB's power spectra for simulation is performed by
           a PySM3 method. We simply provide it with the filepath.
        """
        # Read header line, remove the leading "#"
        with open(path, 'r') as file:
            header_line = file.readline().strip().lstrip('#').split()

        # Read the data into a DataFrame, setting the header manually.
        df = pd.read_csv(path, 
                         comment='#', 
                         sep='\s+', 
                         header=None, 
                         skiprows=1, 
                         names=header_line)

        df = add_missing_multipoles(df, path.name)

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


class PandasCAMBPowerSpectrum(GenericHandler):
    def read(self, path: Path) -> None:
        """
        Method used to read CAMB's power spectra for analysis.

        Reading CAMB's power spectra for simulation is performed by
           a PySM3 method. We simply provide it with the filepath.
        """
        # Read header line, remove the leading "#"
        with open(path, 'r') as file:
            header_line = file.readline().strip().lstrip('#').split()

        # Read the data into a DataFrame, setting the header manually.
        df = pd.read_csv(path, 
                         comment='#', 
                         sep='\s+', 
                         header=None, 
                         skiprows=1, 
                         names=header_line)
        return df

    def write(self, path: Path, data: pd.DataFrame) -> None:
        data_str = df_to_camb_text(data)

        make_directories(path)
        with open(path, "w") as f:
            f.write(data_str)


def df_to_camb_text(df: pd.DataFrame) -> str:
    # Build header line
    header = "#"
    if df.columns[0] != "L":
        raise ValueError("First column in output spectra is not L.")
    header += f"{'L':>3s} "  # L right-aligned in 5, then one space
    for col in df.columns[1:]:
        header += f"{col:<14s}"  # names left-aligned in width 14

    # Build rows
    lines = []
    for _, row in df.iterrows():
        # L as integer, width 4
        l_str = f"{int(row['L']):4d}"

        # Rest as 7-decimal scientific notation
        vals_str = "".join(f"{val: .7e}" for val in row.drop('L'))
        lines.append(f"{l_str}{vals_str}")

    return "\n".join([header] + lines)


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
register_handler("PandasCAMBPowerSpectrum", PandasCAMBPowerSpectrum)
