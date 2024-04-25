from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Detector:
    def __init__(self, nom_freq: int, cen_freq, fwhm):
        self.nom_freq = nom_freq
        self.cen_freq = cen_freq
        self.fwhm = fwhm


def make_detector(bandpass, band):
        # table = self.read_instr_table()
    table = bandpass
    
    band_str = str(band)
    try:
        assert band_str in table['band']
    except AssertionError:
        raise KeyError(f"A detector specified in the configs, {band} " \
                        f"(converted to {band_str}) does not exist in " \
                        f"the given QTable path.")
    
    center_frequency = table.loc[band_str]['center_frequency']
    fwhm = table.loc[band_str]['fwhm']
    return Detector(nom_freq=band, cen_freq=center_frequency, fwhm=fwhm)