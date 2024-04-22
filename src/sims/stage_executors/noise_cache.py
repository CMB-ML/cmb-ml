from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Detector:
    def __init__(self, nom_freq: int, cen_freq, fwhm):
        self.nom_freq = nom_freq
        self.cen_freq = cen_freq
        self.fwhm = fwhm


