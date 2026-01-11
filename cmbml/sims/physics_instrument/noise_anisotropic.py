import logging
import pysm3.units as u
import healpy as hp

from cmbml.sims.physics_instrument.registry_noise import register_noise
from cmbml.sims.physics_instrument.make_noise_scale import make_random_noise_map
from cmbml.sims.physics_instrument.make_noise_scale import ScaleCacheMaker
from cmbml.utils.planck_instrument import Detector
from cmbml.core.config_helper import ConfigHelper

logger = logging.getLogger(__name__)


class NoiseAnisotropicCore:
    """
    Physics: simple variance-driven noise.
    Stateless: operates only on arrays + seed.
    """

    def __init__(self, nside_out: int):
        self.nside_out = nside_out

    def get_noise_map(self, seed: int, sd_map) -> u.Quantity:
        """
        Generate a noise map from a scale-dependent stddev map.

        Parameters
        ----------
        seed : int
            Random seed for reproducibility.
        sd_map : Quantity
            Standard deviation map (already read from an Asset).

        Returns
        -------
        np.ndarray
            Noise map with some nside.
        """
        if hp.get_nside(sd_map) != self.nside_out:
            raise ValueError(f"Input sd_map should have Nside {self.nside_out}")
        return make_random_noise_map(sd_map, seed)


@register_noise("anisotropic")
class NoiseAnisotropic:
    """
    Config-driven wrapper for NoiseAnisotropicCore.
    Responsible for asset reads and name_tracker.
    """
    cache_maker = ScaleCacheMaker

    def __init__(self, cfg, name_tracker, stage_str=None):
        """
        Parameters
        ----------
        cfg : DictConfig
            Hydra configuration object with simulation parameters.
        name_tracker : NameTracker
            Used to manage context for asset resolution and logging.
        scale_cache : Asset
            Provides scale-dependent standard deviation maps.
        """
        self.nside_out = cfg.scenario.nside
        self.unit = u.Unit(cfg.scenario.units)
        self.name_tracker = name_tracker  # Needed for context when lazy-loading

        # Asset handlers
        _ch = ConfigHelper(cfg, "make_noise")
        assets_in = _ch.get_assets_in(name_tracker=self.name_tracker)
        self.in_scale_cache = assets_in["scale_cache"]

        # Caches (to be lazy-loaded)
        self._sd_maps: dict[int, u.Quantity] = {}

        # Core
        self.core = NoiseAnisotropicCore(nside_out=self.nside_out)

    def _lazy_load_freq(self, detector: Detector) -> None:
        """
        Ensure sd maps for a given frequency are loaded and cached.

        Parameters
        ----------
        detector : Detector
            Instrument detector with nominal frequency and fields.
        """
        if detector.nom_freq in self._sd_maps:
            return
        logger.info(f"Lazy-loading noise assets for freq {detector.nom_freq}")
        with self.name_tracker.set_context("freq", detector.nom_freq):
            sd_map: u.Quantity = self.in_scale_cache.read(map_field_strs=detector.fields)
        if isinstance(sd_map, u.Quantity) and sd_map.unit != self.unit:
            eq = u.cmb_equivalencies(detector.cen_freq)
            sd_map = sd_map.to(self.unit, equivalencies=eq)
        self._sd_maps[detector.nom_freq] = sd_map

    def get_noise_map(self, detector: Detector, seed: int) -> u.Quantity:
        self._lazy_load_freq(detector)
        return self.core.get_noise_map(seed, self._sd_maps[detector.nom_freq])


@register_noise("anisotropic_manual")
class NoiseAnisotropicManual:
    """
    Manual wrapper for NoiseAnisotropicCore.

    This class avoids all config and asset logic. The caller must supply
    all required data arrays explicitly on each call. Useful for testing,
    prototyping, and notebooks.
    """

    def __init__(self, nside_out: int):
        """
        Parameters
        ----------
        nside_out : int
            HEALPix nside for the output maps.
        """
        self.core = NoiseAnisotropicCore(nside_out)

    def get_noise_map(self, seed: int, sd_map) -> u.Quantity:
        """
        Generate a noise map directly from supplied arrays.

        Parameters
        ----------
        seed : int
            Random seed for reproducibility.
        sd_map : Quantity
            Standard-deviation map.

        Returns
        -------
        Quantity
            Healpix map containing anisotropic, uncorrelated noise in the 
            appropriate units.
        """
        return self.core.get_noise_map(seed, sd_map)
