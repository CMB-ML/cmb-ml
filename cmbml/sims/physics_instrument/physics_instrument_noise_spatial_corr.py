import logging

import numpy as np
import healpy as hp
import pysm3.units as u
from astropy.units import Quantity

from cmbml.utils.planck_instrument import Detector
from cmbml.sims.physics_instrument.physics_scale_cache_maker import ScaleCacheMaker, make_random_noise_map
from cmbml.utils.physics_downgrade_by_alm import downgrade_by_alm
from cmbml.core.config_helper import ConfigHelper


logger = logging.getLogger(__name__)


class SpatialCorrNoise:
    do_cache = True
    cache_maker = ScaleCacheMaker
    def __init__(self, cfg, name_tracker, scale_cache):
        self.nside_out = cfg.scenario.nside
        self.name_tracker = name_tracker
        self.in_scale_cache = scale_cache

        # Use a ConfigHelper to get the assets in; we don't need them as arguments.
        _ch = ConfigHelper(cfg, 'make_noise')
        assets_in = _ch.get_assets_in(name_tracker=self.name_tracker)
        self.in_noise_model = assets_in["noise_model"]
        self.in_noise_avg   = assets_in["noise_avg"]

        self.map_fields = cfg.scenario.map_fields
        if len(self.map_fields) != 1:
            raise NotImplementedError("Only single field maps are supported for now.")

        lmax_ratio = cfg.model.sim.noise.lmax_ratio_out_noise
        self.lmax_out = int(lmax_ratio * self.nside_out)

        self.avg_maps = {}
        self.n_planck_noise_sims = cfg.model.sim.noise.n_planck_noise_sims

    def load_avg_maps(self, freq):
        if freq in self.avg_maps:
            return
        logger.info(f"Downgrading average map for frequency {freq}, this may take a moment (once per frequency).")
        context = dict(fields=self.map_fields)
        with self.name_tracker.set_contexts(context):
            full_res_avg_map = self.in_noise_avg.read(map_field_strs=self.map_fields)
        self.avg_maps[freq] = downgrade_by_alm(full_res_avg_map, self.nside_out)

    def get_noise_map(self, detector: Detector, noise_seed):
        """
        Returns a noise map for the given frequency and field.
        Called externally in E_make_simulations.py

        Args:
            freq (int): The frequency of the map.
            noise_seed (int): The seed for the noise map.
        """
        freq = detector.nom_freq
        context = dict(freq=freq, n_sims=self.n_planck_noise_sims)
        with self.name_tracker.set_contexts(context):
            self.load_avg_maps(freq)
            sd_map = self.in_scale_cache.read()
            target_cl, tgt_unit = self.get_noise_ps_and_unit(noise_seed)

            wht_noise_map = make_random_noise_map(sd_map, noise_seed)

            if str(tgt_unit) != str(wht_noise_map.unit):
                raise ValueError(f"Target unit {tgt_unit} does not match noise map unit {wht_noise_map.unit}")

            noise_map = correlate_noise(wht_noise_map, target_cl, 
                                        nside=self.nside_out, 
                                        lmax=self.lmax_out)
            noise_map = remove_monopole(noise_map)
            noise_map = add_average_map(noise_map, self.avg_maps[freq])

            return noise_map

    def get_noise_ps_and_unit(self, noise_seed):
        """
        Get the target cls for the given frequency.
        """
        noise_model_fn  = self.in_noise_model.path
        noise_model     = np.load(noise_model_fn)
        src_mean_ps     = noise_model['mean_ps']
        src_components  = noise_model['components']
        src_variance    = noise_model['variance']

        target_cl       = get_target_cls_from_pca_results(1, 
                                                          src_mean_ps, 
                                                          src_variance, 
                                                          src_components,
                                                          noise_seed)

        src_map_unit    = noise_model['maps_unit']
        return target_cl, src_map_unit


def get_target_cls_from_pca_results(n_sims, src_mean_ps, src_variance, src_components, random_seed):
    num_components = len(src_variance)

    std_devs = np.sqrt(src_variance)

    if n_sims == 1:
        reduced_shape = (num_components,)
    else:
        reduced_shape = (n_sims, num_components)

    rng = np.random.default_rng(random_seed)

    reduced_samples = rng.normal(0, std_devs, reduced_shape)
    # Reconstruct power spectra in log10 space
    tgt_log_ps = reduced_samples @ src_components + src_mean_ps
    # Convert out of log10 space
    tgt_cls = 10**tgt_log_ps
    return tgt_cls


def correlate_noise(white_map, target_cl, nside, lmax):
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
    map_unit    = white_map.unit
    white_alms  = hp.map2alm(white_map, lmax=lmax)
    white_cl    = hp.alm2cl(white_alms)
    this_filter = np.sqrt(target_cl[:lmax+1] / white_cl[:lmax+1])
    out_alms    = hp.almxfl(white_alms, this_filter)
    out_map     = hp.alm2map(out_alms, nside=nside)
    out_map     = Quantity(out_map, unit=map_unit)
    return out_map


def remove_monopole(noise_map):
    return noise_map - np.mean(noise_map)

def add_average_map(noise_map, avg_map):
    return noise_map + avg_map
