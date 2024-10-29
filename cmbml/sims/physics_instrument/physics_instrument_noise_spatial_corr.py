import logging

import numpy as np
import healpy as hp
import pysm3.units as u
from astropy.units import Quantity

from cmbml.sims.physics_instrument.physics_scale_cache_maker import ScaleCacheMaker, make_random_noise_map
from cmbml.core.config_helper import ConfigHelper


logger = logging.getLogger(__name__)


class SpatialCorrNoise:
    do_cache = True
    cache_maker = ScaleCacheMaker
    # This class is used to generate noise maps from Planck's observation maps.
    # It is used in the NoiseCacheExecutor and SimCreatorExecutor classes.
    # This class is glue code. The functions afterwards are relevant to Physics.
    def __init__(self, cfg, name_tracker, scale_cache):
        self.nside_out = cfg.scenario.nside
        self.name_tracker = name_tracker
        self.in_scale_cache = scale_cache

        # Use a ConfigHelper to get the assets in; we don't need them as arguments.
        _ch = ConfigHelper(cfg, 'make_noise_cache')  # Applicable to both
                                                     #   NoiseCache and SimCreator
        assets_in = _ch.get_assets_in(name_tracker=self.name_tracker)
        self.in_ps_src     = assets_in["noise_src_cls"]

        self.lmax_out = int(cfg.model.sim.pysm_beam_lmax_ratio * self.nside_out)
        self.boxcar_length = cfg.model.sim.noise.boxcar_length
        self.smooth_initial = cfg.model.sim.noise.smooth_initial

        self.src_root = cfg.local_system.assets_dir
        self.source_ps_fns = cfg.model.sim.noise.src_ps

    def check_target_cl_length(self, freq):
        target_cl = self.get_target_cl(freq)
        if len(target_cl) < self.lmax_out:
            raise ValueError(f"Target Cls must at least length {self.lmax_out}.")

    def get_target_cl(self, freq):
        fn       = self.source_ps_fns[freq]
        contexts_dict = dict(src_root=self.src_root, filename=fn)
        with self.name_tracker.set_contexts(contexts_dict):
            ps = self.in_ps_src.read()
        return ps[:self.lmax_out+1]

    def get_noise_map(self, freq, noise_seed):
        """
        Returns a noise map for the given frequency and field.
        Called externally in E_make_simulations.py

        Args:
            freq (int): The frequency of the map.
            noise_seed (int): The seed for the noise map.
        """
        with self.name_tracker.set_context('freq', freq):
            sd_map = self.in_scale_cache.read()
            wht_noise_map = make_random_noise_map(sd_map, noise_seed)
            
            # Convert to uK_CMB to match loaded target cls
            wht_noise_map = wht_noise_map.to(u.uK_CMB)
            target_cl = self.get_target_cl(freq)
            noise_map = correlate_noise(wht_noise_map, target_cl, 
                                        nside=self.nside_out, 
                                        lmax=self.lmax_out,
                                        boxcar_length=self.boxcar_length, 
                                        smooth_initial=self.smooth_initial)
            return noise_map


def correlate_noise(white_map, target_cl, nside, lmax, boxcar_length, smooth_initial):
    """
    Correlate the noise map with the target_cls.

    Args:
        noise_map (Quantity): The noise map, with astropy units.
        target_cls (np.ndarray): The target cls.
        boxcar_length (int): The boxcar length.
        smooth_initial (int): The number of initial cls to add to the average. 
                              Helps prevent spurious low-ell power.

    Returns:
        np.ndarray: The correlated noise map.
    """
    map_unit = white_map.unit
    white_alms = hp.map2alm(white_map, lmax=lmax)
    white_cls = hp.alm2cl(white_alms)
    this_filter = make_filter(boxcar_length, target_cl, white_cls, smooth_initial)
    out_alms = hp.almxfl(white_alms, this_filter)
    out_map = hp.alm2map(out_alms, nside=nside)
    out_map = Quantity(out_map, unit=map_unit)
    return out_map


def make_filter(boxcar_length, target_cl, source_cl, smooth_initial):
    source_cl = source_cl.copy()
    if boxcar_length == 1:
        return np.sqrt(target_cl / source_cl)
    for i in range(smooth_initial):
        # Noise is white and should be ~constant.
        #    Sometimes the white noise may have very low values in the first few ell bins.
        #    This can cause the filter to be very large in those bins, causing spurious low-ell power.
        #    I'm not sure if it would be better to average these (biased low) or add them (biased high); 
        #    either way these are generally log-scale and we get more reasonable values.
        # source_cl[i] = np.mean(source_cl[i], source_cl.mean())
        source_cl[i] = source_cl[i] + source_cl.mean()
    f = np.sqrt(target_cl / source_cl)
    boxcar = np.ones(boxcar_length)
    f = np.pad(f, (boxcar_length // 2, boxcar_length // 2), mode='edge')
    f = np.convolve(f, boxcar, mode='same') / np.sum(boxcar)
    return f