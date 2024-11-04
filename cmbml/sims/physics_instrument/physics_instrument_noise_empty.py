import logging

import numpy as np
import healpy as hp
# from astropy.units import Unit
import pysm3.units as u
# from astropy.cosmology import Planck15

# import cmbml.utils.fits_inspection as fits_inspect


logger = logging.getLogger(__name__)


class EmptyNoise:
    do_cache = False
    # TODO: Set up abstract class for noise
    # This class returns an array of zeros as a placeholder.
    # It is used in the NoiseCacheExecutor and SimCreatorExecutor classes.
    # This class is glue code. The functions afterwards are relevant to Physics.
    def __init__(self, cfg, *args, **kwargs):
        pass

    def get_noise_map(self, *args, **kwargs):
        """
        Returns an array of zeros as a placeholder.
        Called externally in E_make_simulations.py

        Args:
            freq (int): The frequency of the map.
            field_str (str): The field of the map.
            noise_seed (int): The seed for the noise map.
            center_frequency (float): The center frequency of the detector.
        """
        noise_map = u.Quantity(np.array([0]), u.K_CMB, copy=False)
        return noise_map
