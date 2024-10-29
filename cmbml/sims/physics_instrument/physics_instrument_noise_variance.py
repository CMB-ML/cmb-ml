import logging

from cmbml.sims.physics_instrument.physics_scale_cache_maker import ScaleCacheMaker, make_random_noise_map
from cmbml.core.config_helper import ConfigHelper


logger = logging.getLogger(__name__)


class VarianceNoise:
    do_cache = True
    cache_maker = ScaleCacheMaker
    # This class is used to generate noise maps from Planck's observation maps.
    # It is used in the NoiseCacheExecutor and SimCreatorExecutor classes.
    # This class is glue code. The functions afterwards are relevant to Physics.
    def __init__(self, cfg, name_tracker, scale_cache, in_varmap_src=None):
        self.nside_out = cfg.scenario.nside
        self.name_tracker = name_tracker
        self.in_scale_cache = scale_cache

        # Use a ConfigHelper to get the assets in; we don't need them as arguments.
        _ch = ConfigHelper(cfg, 'make_noise_cache')  # Applicable to both
                                                     #   NoiseCache and SimCreator
        assets_in = _ch.get_assets_in(name_tracker=self.name_tracker)
        self.in_varmap_src = assets_in["noise_src_varmaps"]

        self.in_varmap_src = in_varmap_src

    def get_noise_map(self, freq, noise_seed):
        """
        Returns a noise map for the given frequency and field.
        Called externally in E_make_simulations.py

        Args:
            freq (int): The frequency of the map.
            field_str (str): The field of the map.
            noise_seed (int): The seed for the noise map.
        """
        with self.name_tracker.set_context('freq', freq):
            sd_map = self.in_scale_cache.read()
            noise_map = make_random_noise_map(sd_map, noise_seed)
            return noise_map

