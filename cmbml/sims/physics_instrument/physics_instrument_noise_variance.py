import logging

from cmbml.sims.physics_instrument.physics_scale_cache_maker import ScaleCacheMaker, make_random_noise_map


logger = logging.getLogger(__name__)


class VarianceNoise:
    do_cache = True
    cache_maker = ScaleCacheMaker
    # This class is used to generate noise maps from Planck's observation maps.
    # It is used in the NoiseCacheExecutor and SimCreatorExecutor classes.
    # This class is glue code. The functions afterwards are relevant to Physics.
    def __init__(self, cfg, name_tracker, asset_cache, asset_src=None):
        self.nside_out = cfg.scenario.nside
        self.name_tracker = name_tracker
        self.asset_noise_cache = asset_cache
        self.asset_noise_src = asset_src

    def get_noise_map(self, freq, field_str, noise_seed, center_frequency=None):
        # TODO: Why is center_frequency included?
        """
        Returns a noise map for the given frequency and field.
        Called externally in E_make_simulations.py

        Args:
            freq (int): The frequency of the map.
            field_str (str): The field of the map.
            noise_seed (int): The seed for the noise map.
            center_frequency (float): The center frequency of the detector.
        """
        with self.name_tracker.set_context('freq', freq):
            with self.name_tracker.set_context('field', field_str):
                sd_map = self.asset_noise_cache.read()
                noise_map = make_random_noise_map(sd_map, noise_seed, center_frequency)
                return noise_map

