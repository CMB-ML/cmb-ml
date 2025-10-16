import logging
import numpy as np
import healpy as hp
import pysm3.units as u

from cmbml.sims.physics_instrument.registry_noise import register_noise
from cmbml.sims.physics_instrument.make_noise_scale import make_random_noise_map
from cmbml.sims.physics_instrument.make_noise_scale import ScaleCacheMaker
from cmbml.utils.physics_downgrade_by_alm import downgrade_noise_by_alm
from cmbml.utils.planck_instrument import Detector
from cmbml.core.config_helper import ConfigHelper
from cmbml.core.asset_handlers import HealpyMap


logger = logging.getLogger(__name__)


class NoiseCorrelatedCore:
    """
    Physics: spatially correlated noise with target Cls.

    Stateless: no assets, no caching. Every call requires
    noise_model, sd_map to be provided. If using stationary
    noise, it also requires avg_map to be provided.
    """

    def __init__(self, 
                 nside_out: int, 
                 lmax_out: int, 
                 map_fields: list[str],
                 half_mission: bool=False):
        self.nside_out = nside_out
        self.lmax_out = lmax_out
        self.map_fields = map_fields
        self.half_mission  = half_mission

    def get_noise_map(
        self,
        seed: int,
        noise_model: dict,
        sd_map: u.Quantity,
        avg_map: u.Quantity=None
    ) -> u.Quantity:
        """
        Generate a correlated noise map.

        Parameters
        ----------
        seed : int
            Random seed.
        noise_model : dict
            PCA-style model with keys:
              - 'mean_ps'
              - 'variance'
              - 'components'
              - 'maps_unit'
        sd_map : Quantity
            Standard-deviation map (already read from scale_cache).
        avg_map : Quantity or None
            Average noise map (already downgraded to target nside).

        Returns
        -------
        Quantity
            Healpix map with correlated noise.
        """
        # Checks
        if hp.get_nside(sd_map) != self.nside_out:
            raise ValueError(f"Input sd_map should have Nside {self.nside_out}")
        if avg_map is not None and hp.get_nside(avg_map) != self.nside_out:
            raise ValueError(f"Input avg_map should have Nside {self.nside_out}")

        # Sample target Cls
        target_cl, tgt_unit = self._sample_target_cls(noise_model, seed)
        if self.half_mission:
            target_cl *= 2

        # White noise map
        white_map = make_random_noise_map(sd_map, seed)

        if str(tgt_unit) != str(white_map.unit):
            raise ValueError(f"Unit mismatch: {tgt_unit} vs {white_map.unit}")

        # Correlate with filter
        white_alms = hp.map2alm(white_map, lmax=self.lmax_out)
        white_cl = hp.alm2cl(white_alms)
        filt = np.sqrt(target_cl[: self.lmax_out + 1] / white_cl[: self.lmax_out + 1])
        out_alms = hp.almxfl(white_alms, filt)
        out_map = hp.alm2map(out_alms, nside=self.nside_out)
        out_map = u.Quantity(out_map, unit=white_map.unit)

        if avg_map is not None:
            # Mean-center and add average map
            out_map = out_map - np.mean(out_map) + avg_map
        return out_map

    def _sample_target_cls(self, noise_model: dict, seed: int):
        """Reconstruct a target Cl using PCA-style model."""
        src_mean_ps = noise_model["mean_ps"]
        src_components = noise_model["components"]
        src_variance = noise_model["variance"]
        src_map_unit = noise_model["maps_unit"]

        rng = np.random.default_rng(seed)
        reduced_samples = rng.normal(0, np.sqrt(src_variance))
        tgt_log_ps = reduced_samples @ src_components + src_mean_ps
        tgt_cls = 10 ** tgt_log_ps

        return tgt_cls, src_map_unit


@register_noise("stationary")
class NoiseStationary:
    """
    Config-driven wrapper for NoiseCorrelatedCore. This produces noise
    with spatial correlation and stationary components (via average map).

    This class integrates with the simulation pipeline: it reads assets
    (noise models, average maps, and standard-deviation maps) using Hydra
    configs and the name tracker, and lazily caches them by frequency. On
    subsequent calls for the same frequency, cached data is reused.

    Use this in executors or other config-based workflows where asset
    handlers and lazy-loading are expected.
    """
    cache_maker = ScaleCacheMaker

    def __init__(self, 
                 cfg, 
                 name_tracker, 
                 half_mission:bool=False):
        """
        Parameters
        ----------
        cfg : DictConfig
            Hydra configuration object with simulation parameters.
        name_tracker : NameTracker
            Used to manage context for asset resolution and logging.
        """
        self.nside_out = cfg.scenario.nside
        self.unit = u.Unit(cfg.scenario.units)
        self.lmax_out = int(cfg.model.sim.noise.lmax_ratio_out_noise * cfg.scenario.nside)
        self.map_fields = cfg.scenario.map_fields
        self.name_tracker = name_tracker
        self.n_planck_noise_sims = cfg.model.sim.noise.n_planck_noise_sims

        # Asset handlers
        _ch = ConfigHelper(cfg, "make_noise")
        assets_in = _ch.get_assets_in(name_tracker=self.name_tracker)
        self.in_noise_model = assets_in["noise_model"]
        self.in_scale_cache = assets_in["scale_cache"]
        self.in_noise_avg = assets_in["noise_avg"]
        in_map_handler = HealpyMap

        # Caches (to be lazy-loaded)
        self._noise_models: dict[int, dict] = {}
        self._sd_maps:  dict[int, u.Quantity] = {}
        self._avg_maps: dict[int, u.Quantity] = {}

        # Core
        self.core = NoiseCorrelatedCore(
            nside_out=self.nside_out,
            lmax_out=self.lmax_out,
            map_fields=self.map_fields,
            half_mission=half_mission
        )

    def _lazy_load_freq(self, detector: Detector):
        """
        Ensure noise model, standard-deviation map, and average map for a 
        given frequency are loaded and cached.

        Parameters
        ----------
        detector : Detector
            Instrument detector with nominal frequency and fields.
        """
        if detector.nom_freq in self._sd_maps:
            return
        logger.info(f"Lazy-loading noise assets for freq {detector.nom_freq}")
        context = dict(
            n_sims=self.n_planck_noise_sims,
            freq=detector.nom_freq,
            fields="I"  # Kludge for now
        )
        with self.name_tracker.set_contexts(context):
            noise_model = np.load(self.in_noise_model.path, allow_pickle=True)
            nm_unit = u.Unit(str(noise_model["maps_unit"]))
            if nm_unit != self.unit:
                raise ValueError("Unit provided does not match noise model.")
            sd_map = self.in_scale_cache.read(map_field_strs=detector.fields)
            avg_map = self.in_noise_avg.read(map_field_strs=self.map_fields)
            avg_map = downgrade_noise_by_alm(avg_map, self.nside_out)

        if isinstance(sd_map, u.Quantity) and sd_map.unit != self.unit:
            eq = u.cmb_equivalencies(detector.cen_freq)
            sd_map = sd_map.to(self.unit, equivalencies=eq)
        if isinstance(avg_map, u.Quantity) and avg_map.unit != self.unit:
            eq = u.cmb_equivalencies(detector.cen_freq)
            avg_map = avg_map.to(self.unit, equivalencies=eq)

        self._noise_models[detector.nom_freq] = noise_model
        self._sd_maps[detector.nom_freq] = sd_map
        self._avg_maps[detector.nom_freq] = avg_map

    def get_noise_map(self, detector: Detector, seed: int):
        """
        Generate a noise map for a detector and seed, loading assets if needed.

        Parameters
        ----------
        detector : Detector
            Instrument detector with nominal frequency and fields.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        Quantity
            Healpix map containing correlated noise in the appropriate units.
        """
        self._lazy_load_freq(detector)
        freq = detector.nom_freq
        with self.name_tracker.set_context("freq", freq):
            return self.core.get_noise_map(
                seed,
                self._noise_models[freq],
                self._sd_maps[freq],
                self._avg_maps[freq]
            )


@register_noise("stationary_manual")
class NoiseStationaryManual:
    """
    Manual wrapper for NoiseCorrelatedCore. This produces noise
    with spatial correlation and stationary components (via average map).

    This class avoids all config and asset logic. The caller must supply
    all required data arrays explicitly on each call. Useful for testing,
    prototyping, and notebooks.
    """
    def __init__(self, 
                 nside_out: int, 
                 lmax_out: int, 
                 map_fields: list[str],
                 half_mission: bool=False
                 ):
        """
        Parameters
        ----------
        nside_out : int
            HEALPix nside for the output maps.
        lmax_out : int
            Maximum multipole used in map generation.
        map_fields : list of str
            Fields of the map (e.g., ['I_STOKES'], ['Q_STOKES','U_STOKES']).
        """
        self.nside_out = nside_out
        self.core = NoiseCorrelatedCore(nside_out, lmax_out, map_fields, half_mission=half_mission)

    def get_noise_map(self, 
                      seed: int, 
                      noise_model, 
                      sd_map, 
                      avg_map):
        """
        Generate a noise map directly from supplied arrays.

        Parameters
        ----------
        seed : int
            Random seed for reproducibility.
        noise_model : dict
            Dictionary containing PCA-style model with keys:
            - 'mean_ps'
            - 'variance'
            - 'components'
            - 'maps_unit'
        sd_map : Quantity
            Standard-deviation map.
        avg_map : Quantity
            Average noise map, already downgraded to the target nside.

        Returns
        -------
        Quantity
            Healpix map containing correlated noise in the appropriate units.
        """
        if hp.get_nside(avg_map) != self.nside_out:
            avg_map = downgrade_noise_by_alm(avg_map, self.nside_out)
        return self.core.get_noise_map(seed, noise_model, sd_map, avg_map)


@register_noise("correlated")
class NoiseCorrelated:
    """
    Config-driven wrapper for NoiseCorrelatedCore. This produces noise
    with spatial correlation, but no stationary components.

    This class integrates with the simulation pipeline: it reads assets
    (noise models and standard-deviation maps) using Hydra configs
    and the name tracker, and lazily caches them by frequency. On
    subsequent calls for the same frequency, cached data is reused.

    Use this in executors or other config-based workflows where asset
    handlers and lazy-loading are expected.
    """
    cache_maker = ScaleCacheMaker

    def __init__(self, cfg, name_tracker, half_mission:bool=False):
        """
        Parameters
        ----------
        cfg : DictConfig
            Hydra configuration object with simulation parameters.
        name_tracker : NameTracker
            Used to manage context for asset resolution and logging.
        """
        self.nside_out = cfg.scenario.nside
        self.unit = u.Unit(cfg.scenario.units)
        self.lmax_out = int(cfg.model.sim.noise.lmax_ratio_out_noise * cfg.scenario.nside)
        self.map_fields = cfg.scenario.map_fields
        self.name_tracker = name_tracker
        self.n_planck_noise_sims = cfg.model.sim.noise.n_planck_noise_sims

        # Asset handlers
        _ch = ConfigHelper(cfg, "make_noise")
        assets_in = _ch.get_assets_in(name_tracker=self.name_tracker)
        self.in_noise_model = assets_in["noise_model"]
        self.in_scale_cache = assets_in["scale_cache"]

        # Caches (to be lazy-loaded)
        self._noise_models: dict[int, dict] = {}
        self._sd_maps:  dict[int, u.Quantity] = {}

        # Core
        self.core = NoiseCorrelatedCore(
            nside_out=self.nside_out,
            lmax_out=self.lmax_out,
            map_fields=self.map_fields,
            half_mission=half_mission
        )

    def _lazy_load_freq(self, detector: Detector):
        """
        Ensure noise model and standard-deviation maps for a given frequency
        are loaded and cached.

        Parameters
        ----------
        detector : Detector
            Instrument detector with nominal frequency and fields.
        """
        if detector.nom_freq in self._sd_maps:
            return
        logger.info(f"Lazy-loading noise assets for freq {detector.nom_freq}")
        context = dict(
            n_sims=self.n_planck_noise_sims,
            freq=detector.nom_freq
        )
        with self.name_tracker.set_contexts(context):
            noise_model = np.load(self.in_noise_model.path, allow_pickle=True)
            nm_unit = u.Unit(str(noise_model["maps_unit"]))
            if nm_unit != self.unit:
                raise ValueError(f"Noise model unit {nm_unit} != expected {self.unit}")
            sd_map = self.in_scale_cache.read(map_field_strs=detector.fields)
        if isinstance(sd_map, u.Quantity) and sd_map.unit != self.unit:
            eq = u.cmb_equivalencies(detector.cen_freq)
            sd_map = sd_map.to(self.unit, equivalencies=eq)
        self._noise_models[detector.nom_freq] = noise_model
        self._sd_maps[detector.nom_freq] = sd_map

    def get_noise_map(self, detector: Detector, seed: int):
        """
        Generate a noise map for a detector and seed, loading assets if needed.

        Parameters
        ----------
        detector : Detector
            Instrument detector with nominal frequency and fields.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        Quantity
            Healpix map containing correlated noise in the appropriate units.
        """
        self._lazy_load_freq(detector)
        freq = detector.nom_freq
        with self.name_tracker.set_context("freq", freq):
            return self.core.get_noise_map(
                seed,
                self._noise_models[freq],
                self._sd_maps[freq],
            )


@register_noise("correlated_manual")
class NoiseCorrelatedManual:
    """
    Manual wrapper for NoiseCorrelatedCore. This produces noise
    with spatial correlation, but no stationary components.

    This class avoids all config and asset logic. The caller must supply
    all required data arrays explicitly on each call. Useful for testing,
    prototyping, and notebooks.
    """
    def __init__(self, nside_out: int, lmax_out: int, map_fields: list[str], half_mission:bool=False):
        """
        Parameters
        ----------
        nside_out : int
            HEALPix nside for the output maps.
        lmax_out : int
            Maximum multipole used in map generation.
        map_fields : list of str
            Fields of the map (e.g., ['I_STOKES'], ['Q_STOKES','U_STOKES']).
        """
        self.core = NoiseCorrelatedCore(nside_out, lmax_out, map_fields, half_mission=half_mission)

    def get_noise_map(self, seed: int, noise_model, sd_map):
        """
        Generate a noise map directly from supplied arrays.

        Parameters
        ----------
        seed : int
            Random seed for reproducibility.
        noise_model : dict
            Dictionary containing PCA-style model with keys:
            - 'mean_ps'
            - 'variance'
            - 'components'
            - 'maps_unit'
        sd_map : Quantity
            Standard-deviation map.

        Returns
        -------
        Quantity
            Healpix map containing correlated noise in the appropriate units.
        """
        return self.core.get_noise_map(seed, noise_model, sd_map)
