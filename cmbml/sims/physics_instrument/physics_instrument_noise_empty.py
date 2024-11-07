import logging

import numpy as np
import healpy as hp
# from astropy.units import Unit
import pysm3.units as u
# from astropy.cosmology import Planck15

# import cmbml.utils.fits_inspection as fits_inspect
from cmbml.utils.planck_instrument import Detector


logger = logging.getLogger(__name__)


class EmptyNoise:
    do_cache = False
    # TODO: Set up abstract class for noise
    # This class returns an array of zeros as a placeholder.
    # It is used in the NoiseCacheExecutor and SimCreatorExecutor classes.
    # This class is glue code. The functions afterwards are relevant to Physics.
    def __init__(self, cfg, *args, **kwargs):
        nside = cfg.scenario.nside
        self.npix = hp.nside2npix(nside)

    def get_noise_map(self, detector: Detector, *args, **kwargs):
        """
        Returns an array of zeros as a placeholder.
        Called externally in E_make_simulations.py

        Args:
            freq (int): The frequency of the map.
            field_str (str): The field of the map.
            noise_seed (int): The seed for the noise map.
            center_frequency (float): The center frequency of the detector.
        """
        out_shape = (len(detector.fields), self.npix)
        noise_map = u.Quantity(np.zeros(shape=out_shape), detector.unit, copy=False)
        return noise_map
