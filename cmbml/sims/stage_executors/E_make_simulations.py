from typing import Dict, Union
from pathlib import Path
import logging

import hydra
from omegaconf import DictConfig

import numpy as np
from astropy.units import Quantity
import pysm3
import pysm3.units as u
from pysm3 import CMBLensed
from tqdm import tqdm

from cmbml.sims.cmb_factory import CMBFactory
# from cmbml.sims.random_seed_manager import FieldLevelSeedFactory
from cmbml.sims.random_seed_manager import FreqLevelSeedFactory
from cmbml.sims.random_seed_manager import SimLevelSeedFactory
from cmbml.utils.planck_instrument import make_instrument, Instrument

from cmbml.core import (
    BaseStageExecutor,
    Split,
    Asset, AssetWithPathAlts
)

from cmbml.core.asset_handlers.qtable_handler import QTableHandler # Import to register handler
from cmbml.core.asset_handlers.psmaker_handler import CambPowerSpectrum, NumpyPowerSpectrum # Import for typing hint
from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap # Import for VS Code hints

from cmbml.utils.map_formats import convert_pysm3_to_hp
from cmbml.sims.physics_instrument import get_noise_class

import healpy as hp


logger = logging.getLogger(__name__)


class SimCreatorExecutor(BaseStageExecutor):
    """
    SimCreatorExecutor is responsible for generating the simulated maps for a given simulation scenario.

    Attributes:
        out_cmb_map (Asset): The output asset for the CMB map.
        out_obs_maps (Asset): The output asset for the observation maps.
        in_noise_cache (Asset): The input asset for the noise cache.
        in_cmb_ps (AssetWithPathAlts): The input asset for the CMB power spectra.
        in_det_table (Asset): The input asset for the detector table.
        instrument (Instrument): The instrument configuration used for the simulation.
        cmb_seed_factory (SimLevelSeedFactory): The seed factory for the CMB.
        noise_seed_factory (FieldLevelSeedFactory): The seed factory for the noise.
        nside_sky (int): The nside for the sky.
        nside_out (int): The nside for the output maps.
        lmax_out (int): The lmax for the output maps.
        units (str): The units for the output maps.
        preset_strings (List[str]): The preset strings for the sky.
        output_units (str): The output units for the sky.
        cmb_factory (CMBFactory): The factory for the CMB.
        sky (pysm3.Sky): The sky object for the simulation.

    Methods:
        execute() -> None:
            Executes the simulation generation process.
        process_split(split: Split) -> None:
            Produces all sims for a split.
        process_sim(split: Split, sim_num: int) -> None:
            Processes the given split and simulation number.
    """
    def __init__(self, cfg: DictConfig) -> None:
        # The following stage_str must match the pipeline yaml
        super().__init__(cfg, stage_str='make_sims')

        self.out_cmb_map: Asset = self.assets_out['cmb_map']
        self.out_obs_maps: Asset = self.assets_out['obs_maps']
        self.out_noise_maps: Asset = self.assets_out['noise_maps']
        out_cmb_map_handler: HealpyMap
        out_obs_maps_handler: HealpyMap

        self.in_noise_cache: Asset = self.assets_in['scale_cache']
        self.in_cmb_ps: AssetWithPathAlts = self.assets_in['cmb_ps']
        in_det_table: Asset = self.assets_in['planck_deltabandpass']
        in_noise_cache_handler: Union[HealpyMap, NumpyPowerSpectrum]
        in_cmb_ps_handler: CambPowerSpectrum
        in_det_table_handler: QTableHandler

        # Initialize constants from configs
        self.nside_sky = self.get_nside_sky()
        logger.info(f"Simulations will generated at nside_sky = {self.nside_sky}.")
        self.nside_out = cfg.scenario.nside
        logger.info(f"Simulations will be output at nside_out = {self.nside_out}")
        self.output_units = cfg.scenario.units
        logger.info(f"Output units are {self.output_units}")
        self.preset_strings = None if cfg.model.sim.preset_strings is None else list(cfg.model.sim.preset_strings)
        logger.info(f"Preset strings are {self.preset_strings}")

        # The instrument object contains both
        #   - information about physical detector parameters
        #   - information about configurations, (such as fields to use)
        det_info = in_det_table.read()
        self.instrument: Instrument = make_instrument(cfg=cfg, det_info=det_info)

        self.cmb_seed_factory       = SimLevelSeedFactory(cfg, cfg.model.sim.cmb.seed_string)
        self.cmb_factory = CMBFactory(cfg)

        self.noise_seed_factory     = FreqLevelSeedFactory(cfg, cfg.model.sim.noise.seed_string)
        NoiseMaker                  = get_noise_class(cfg.model.sim.noise.noise_type)
        self.noise_maker            = NoiseMaker(cfg, self.name_tracker, self.in_noise_cache)

        # Saving noise is optional here; noise may be generated and added in another method
        self.save_noise             = cfg.model.sim.noise.save_noise

        # Do not create the Sky object here, it takes too long and will slow down initial checks
        self.sky = None

    def execute(self) -> None:
        """
        Creates simulations for all sims within all splits.

        Sets up the Sky object just once
           - Make placeholder object for CMB (others could be added here)
           - Preset strings are passed here
        Thus, components from preset strings are created here, once, for all simulations
        """
        logger.debug(f"Running {self.__class__.__name__} execute() method")
        placeholder = pysm3.Model(nside=self.nside_sky, max_nside=self.nside_sky)
        logger.debug('Creating PySM3 Sky object')
        self.sky = pysm3.Sky(nside=self.nside_sky,
                             component_objects=[placeholder],
                             preset_strings=self.preset_strings,
                             output_unit=self.output_units)
        logger.debug('Done creating PySM3 Sky object')
        self.default_execute()

    def process_split(self, split: Split) -> None:
        """
        Processes all sims for a split, making simulations.
        Hollow boilerplate.

        Args:
            split (Split): The split to process.
        """
        with tqdm(total=split.n_sims, desc=f"{split.name}: ", leave=False) as pbar:
            for sim in split.iter_sims():
                pbar.set_description(f"{split.name}: {sim:04d}")
                with self.name_tracker.set_context("sim_num", sim):
                    self.process_sim(split, sim_num=sim)
                pbar.update(1)

    def process_sim(self, split: Split, sim_num: int) -> None:
        """
        Produces a single simulation. Expects the sky object to be initialized.

        Args:
            split (Split): The split to process. Needed for some configuration information.
            sim_num (int): The simulation number.
        """
        sim_name = self.name_tracker.sim_name()  # for logging only
        logger.debug(f"Creating simulation {split.name}:{sim_name}")

        # One CMB seed per simulation - constant for all frequencies
        cmb_seed = self.cmb_seed_factory.get_seed(split, sim_num)
        ps_path = self.in_cmb_ps.path_alt if split.ps_fidu_fixed else self.in_cmb_ps.path
        cmb = self.cmb_factory.make_cmb(cmb_seed, ps_path)

        # Replace placeholder CMB (or previous simulation's CMB) with new CMB
        self.sky.components[0] = cmb

        # Track minimum FWHM; this will be used for the CMB map
        min_fwhm = 0 * u.arcmin

        for freq, detector in self.instrument.dets.items():
            skymaps = self.sky.get_emission(detector.cen_freq)

            n_fields_sky = skymaps.shape[0]
            n_fields_det = len(detector.fields)
            if n_fields_sky == n_fields_det:
                pass
            elif n_fields_sky == 3 and n_fields_det == 1:
                # PySM3 components always include T, Q, U; extract the temperature map
                skymaps = skymaps[0]
            # else:  # There may be other cases, but none come to mind.
            #     pass

            # One noise realization per frequency
            noise_seed = self.noise_seed_factory.get_seed(split.name, sim_num, freq)
            noise_map = self.noise_maker.get_noise_map(freq, noise_seed)

            # Use pysm3.apply_smoothing... to convolve the map with the planck detector beam
            map_smoothed = pysm3.apply_smoothing_and_coord_transform(skymaps,
                                                                     detector.fwhm,
                                                                     # let PySM3 decide the lmax. This is appropriate 
                                                                     #    as long as the Nside_sky >= 2*Nside_out 
                                                                     #  lmax=self.lmax_beam,
                                                                     output_nside=self.nside_out)
            final_map = map_smoothed + noise_map

            column_names = []
            for field_str in detector.fields:
                column_names.append(field_str + "_STOKES")
            with self.name_tracker.set_contexts(dict(freq=freq)):
                self.out_obs_maps.write(data=final_map, column_names=column_names)
                if self.save_noise:
                    self.out_noise_maps.write(data=noise_map, column_names=column_names)
            logger.debug(f"For {split.name}:{sim_name}, {freq} GHz: done with channel")

        self.save_cmb_map_realization(cmb, min_fwhm)
        logger.debug(f"For {split.name}:{sim_name}, done with simulation")

    def save_cmb_map_realization(self, cmb: CMBLensed, min_fwhm):
        """
        Saves a realization of the CMB map to the output asset.

        Args:
            cmb (CMBLensed): The CMB object to save.
        """
        cmb_realization: Quantity = cmb.map
        # PySM3 components always include T, Q, U, so we may need to extract the temperature map
        if self.instrument.map_fields == 'I':
            cmb_realization = cmb_realization[0]

        scaled_map = pysm3.apply_smoothing_and_coord_transform(cmb_realization,
                                                               fwhm=min_fwhm,
                                                               # let PySM3 decide the lmax. This is appropriate
                                                               #    as long as the Nside_sky >= 2*Nside_out
                                                               #  lmax=self.lmax_beam,
                                                               output_nside=self.nside_out)
        self.out_cmb_map.write(data=scaled_map)

    def get_nside_sky(self):
        """
        Returns the nside to use for PySM3's sky object. May be set with one of two 
        configuration options.
        """
        nside_out = self.cfg.scenario.nside
        nside_sky_set = self.cfg.model.sim.get("nside_sky", None)
        nside_sky_factor = self.cfg.model.sim.get("nside_sky_factor", None)

        nside_sky = nside_sky_set if nside_sky_set else nside_out * nside_sky_factor
        return nside_sky
