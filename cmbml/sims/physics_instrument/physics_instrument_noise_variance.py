import logging

from cmbml.sims.physics_instrument.physics_scale_cache_maker import (
    ScaleCacheMaker, 
    make_random_noise_map
    )
from cmbml.core.config_helper import ConfigHelper


logger = logging.getLogger(__name__)


class VarianceNoise:
    do_cache = True
    cache_maker = ScaleCacheMaker
    # This class generates noise maps from Planck's observation maps.
    #    It is used in the NoiseCacheExecutor and SimCreatorExecutor classes.
    #    This class is glue code. The functions relevant to Physics are in 
    #    physics_scale_cache_maker.py.
    def __init__(self, cfg, name_tracker, scale_cache):
        self.nside_out = cfg.scenario.nside
        self.name_tracker = name_tracker
        self.in_scale_cache = scale_cache

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
