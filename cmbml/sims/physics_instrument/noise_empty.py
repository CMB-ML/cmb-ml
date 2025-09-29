import logging
import numpy as np
import healpy as hp
import pysm3.units as u

from cmbml.sims.physics_instrument.registry_noise import register_noise
from cmbml.sims.physics_instrument.make_noise_scale import make_random_noise_map
from cmbml.utils.planck_instrument import Detector
from cmbml.core.config_helper import ConfigHelper


logger = logging.getLogger(__name__)


class NoiseNoneCore:
    """
    Physics: zero vector of noise.
    Stateless: operates only on arrays + seed.
    """

    def __init__(self, nside_out: int):
        self.nside_out = nside_out

    def get_noise_map(self, detector: Detector=None) -> u.Quantity:
        """
        Generate a noise map from a pre-read scale-dependent stddev map.

        Returns
        -------
        np.ndarray
            Healpix noise map.
        """
        if detector is None:
            n_fields = 1
        else:
            n_fields = len(detector.fields)

        out_shape = (n_fields, hp.nside2npix(self.nside_out))
        m = np.zeros(shape=out_shape)
        return m


@register_noise("noise_empty")
class NoiseEmpty:
    """
    Config-driven wrapper for NoiseNoneCore.
    Responsible for asset reads and name_tracker.
    """
    def __init__(self, cfg, *args, **kwargs):
        """
        Parameters
        ----------
        cfg : DictConfig
            Hydra configuration object with simulation parameters.
        args: list
            Content to be dumped. This is just a common interface.
        kwargs : dict
            Content to be dumped.
        """
        self.nside_out = cfg.scenario.nside
        self.unit = u.Unit(cfg.scenario.units)

        # Core
        self.core = NoiseNoneCore(nside_out=self.nside_out)

    def get_noise_map(self, detector=None, *args, **kwargs) -> u.Quantity:
        m = self.core.get_noise_map(detector)
        if self.unit is not None:
            m = u.Quantity(m, self.unit)
        return m


@register_noise("noise_empty_manual")
class NoiseEmptyManual:
    """
    Manual wrapper for NoiseNoneCore.

    This class avoids all config and asset logic. The caller must supply
    all required data arrays explicitly on each call. Useful for testing,
    prototyping, and notebooks.
    """

    def __init__(self, nside_out: int, unit: u.Unit=None):
        """
        Parameters
        ----------
        nside_out : int
            HEALPix nside for the output maps.
        """
        self.core = NoiseNoneCore(nside_out)
        self.unit = unit

    def get_noise_map(self, detector:Detector=None, **kwargs) -> u.Quantity:
        """
        Return map with all zeros.

        Parameters
        ----------
        kwargs are dumped

        Returns
        -------
        Quantity
            Healpix map with proper units, but zero values throughout.
        """
        m = self.core.get_noise_map(detector)
        if self.unit is not None:
            m = u.Quantity(m, self.unit)
        return m
