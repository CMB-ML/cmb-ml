from typing import Dict

import hydra
from omegaconf import DictConfig
from pathlib import Path

from .sim_stuff import CMBFactory, FieldLevelSeedFactory, SimLevelSeedFactory

import pysm3
import pysm3.units as u
from pysm3 import CMBLensed
import numpy as np
import camb

import logging
from .noise_cache import Detector

from ...core import (
    BaseStageExecutor,
    ExperimentParameters,
    Split,
    Asset
)

from ..physics_cmb import map2ps, convert_to_log_power_spectrum, scale_fiducial_cmb


logger = logging.getLogger(__name__)

class SimCreatorExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig, experiment: ExperimentParameters) -> None:
        self.stage_str = 'create-sims'
        super().__init__(cfg, experiment)

        self.planck_deltabandpass = self.assets_in['planck_deltabandpass']
        
        with self.name_tracker.set_context('src_root', cfg.local_system.noise_src_dir):
            self.deltabandpass = self.planck_deltabandpass.read()
        

        self.in_noise_cache = self.assets_in['noise_cache']
        self.in_wmap_fixed = self.assets_in['wmap_config_fixed']
        self.in_wmap_varied = self.assets_in['wmap_config_varied']
        self.in_ps_fixed = self.assets_in['cmb_ps_fid_fixed']
        self.in_ps_varied = self.assets_in['cmb_ps_fid_varied']

        self.out_cmb_map_fid = self.assets_out['cmb_map_fid']
        self.out_cmb_ps_der = self.assets_out['cmb_ps_der']
        self.out_sims = self.assets_out['sims']

        # seed maker objects
        self.cmb_seed_factory = SimLevelSeedFactory(cfg, 'cmb')
        self.noise_seed_factory = FieldLevelSeedFactory(cfg, 'noise')
        # Initialize constants from configs

        self.nside_out = self.experiment.nside
        self.nside_sky = cfg.simulation.nside_sky
        
        # assert self.nside_sky > self.nside_out, "nside of sky should be greater than nside of target output by at least a factor of 2"

        self.preset_strings = cfg.simulation.preset_strings
        self.planck_freqs = self.experiment.detector_freqs
        self.field_strings = self.experiment.map_fields
        self.lmax_pysm3_smoothing = int(cfg.simulation.cmb.derived_ps_nsmax_x * self.nside_out)
        self.lmax_derived_ps = int(cfg.simulation.cmb.derived_ps_nsmax_x * self.nside_out)

        self.placeholder = pysm3.Model(nside=self.nside_sky, max_nside=self.nside_sky)

        self.sky = pysm3.Sky(nside=self.nside_sky, component_objects=[self.placeholder])
        
        self.cmb_factory = CMBFactory(cfg)

    def execute(self) -> None:
        super().execute()
    
    def process_split(self, split: Split) -> None:
        if split.ps_fidu_fixed:
            logger.debug(f"Working in {split.name}, fixed ps")
            self.process_sims(split, self.in_ps_fixed)
        else:
            logger.debug(f"Working in {split.name}, varied ps")
            self.process_sims(split, self.in_ps_varied)
        
    
    def process_sims(self, split, ps) -> None:
        for sim in split.iter_sims():
            with self.name_tracker.set_context("sim_num", sim):
                cmb_seed = self.cmb_seed_factory.get_seed(split, sim)
                
                

                cmb = self.cmb_factory.make_cmb_lensed(cmb_seed, ps)

                self.sky.components[0] = cmb

                self.save_fid_cmb_map(cmb, self.out_cmb_map_fid)

                self.save_der_cmb_ps(cmb, self.out_cmb_ps_der, lmax=self.lmax_derived_ps)

                for freq in self.planck_freqs:
                    # beam = planck.get_beam(freq)
                    detector = self.make_detector(freq)

                    skymaps = self.sky.get_emission(detector.cen_freq)

                    obs_map = []
                    # print(len(skymaps))
                    for skymap, field_str in zip(skymaps, self.field_strings):
                        if freq in [545, 857] and field_str != "T":
                            continue
                        
                        map_smoothed = pysm3.apply_smoothing_and_coord_transform(skymap, 
                                                                                 detector.fwhm, 
                                                                                 lmax=self.lmax_pysm3_smoothing, 
                                                                                 output_nside=self.nside_out)

                        noise_seed = self.noise_seed_factory.get_seed(split, sim, freq, field_str)
                        noise_map = self.get_noise_map(freq, field_str, noise_seed, detector.cen_freq)
                        final_map = map_smoothed + noise_map
                        obs_map.append(final_map)
                    self.out_sims.write({str(freq): obs_map})

    def make_detector(self, band):
        # table = self.read_instr_table()
        table = self.deltabandpass
        
        band_str = str(band)
        try:
            assert band_str in table['band']
        except AssertionError:
            raise KeyError(f"A detector specified in the configs, {band} " \
                            f"(converted to {band_str}) does not exist in " \
                            f"the QTable ({self.table}).")
        
        center_frequency = table.loc[band_str]['center_frequency']
        fwhm = table.loc[band_str]['fwhm']
        return Detector(nom_freq=band, cen_freq=center_frequency, fwhm=fwhm)

    def save_fid_cmb_map(self, cmb: CMBLensed, asset: Asset):
        fid_cmb_map = cmb.map
        units = fid_cmb_map.unit        # Astropy being all clever and tracking unity
        values = fid_cmb_map.value       # The numbers in the array
        nside_out = self.nside_out
        scaled_map = scale_fiducial_cmb(values, nside_out)
        scaled_map = scaled_map * units
        logger.debug(f"Saving fiducial cmb_map for {asset.path.parent.name}")
        # TODO: Get answer of asset handler healpymap vs sim.write_fid_map
        asset.write(scaled_map)


    def save_der_cmb_ps(self, cmb: CMBLensed, asset: Asset, lmax: int):
        logger.debug(f"Getting derived cmb_ps for {asset.path.parent.name}")
        fid_cmb_map = cmb.map
        ps_cl = map2ps(fid_cmb_map, lmax)
        ps_dl = convert_to_log_power_spectrum(ps_cl)
        logger.debug(f"Saving fiducial cmb_ps for {asset.path.parent.name}")
        camb.results.save_cmb_power_array(asset.path,
                                        array=ps_dl,
                                        labels="TT EE BB TE EB TB")

    def get_noise_map(self, freq, field_str, noise_seed, center_frequency=None):
        with self.name_tracker.set_context('freq', freq):
            with self.name_tracker.set_context('field', field_str):
                logger.debug(f"Getting noise map for {freq} GHz, {field_str}")
                sd_map = self.in_noise_cache.read()
                noise_map = self._make_random_noise_map(sd_map, noise_seed, center_frequency)
                return noise_map

    def _make_random_noise_map(self, sd_map, random_seed, center_frequency=None):
        #TODO: set units when redoing this function
        logger.debug(f'Seed being used: {random_seed}')
        logger.debug(f"physics_instrument_noise.make_random_noise_map start")
        rng = np.random.default_rng(random_seed)
        noise_map = rng.normal(scale=sd_map, size=sd_map.size)
        noise_map = u.Quantity(noise_map, u.K_CMB, copy=False)
        #TODO: take units as output instead of the following line.
        noise_map = noise_map.to(u.uK_RJ, equivalencies=u.cmb_equivalencies(center_frequency))
        logger.debug(f"physics_instrument_noise.make_random_noise_map end")
        return noise_map
        



        
        


