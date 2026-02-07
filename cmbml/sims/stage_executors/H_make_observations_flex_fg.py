from typing import Dict, Union
from pathlib import Path
import logging

import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np
from astropy.units import Quantity
import pysm3
import pysm3.units as u
from pysm3 import CMBLensed
from tqdm import tqdm

from cmbml.sims.cmb_factory import CMBFactory
from cmbml.sims.random_seed_manager import SeedFactory
from cmbml.utils.planck_instrument import make_instrument, Instrument

from cmbml.core import (
    BaseStageExecutor,
    Split,
    Asset, AssetWithPathAlts
)

from cmbml.core.asset_handlers.qtable_handler import QTableHandler # Import to register handler
from cmbml.core.asset_handlers.ps_handler import CambPowerSpectrum, NumpyPowerSpectrum # Import for typing hint
from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap # Import for VS Code hints

from cmbml.utils.map_formats import convert_pysm3_to_hp
from cmbml.utils.pysm_flex_sky import FlexSky

import healpy as hp


logger = logging.getLogger(__name__)


class ObsCreatorExecutor(BaseStageExecutor):
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
    def __init__(self, cfg: DictConfig, stage_str='make_obs_no_noise') -> None:
        # The following stage_str must match the pipeline yaml
        super().__init__(cfg, stage_str=stage_str)

        self.include_cmb = cfg.model.sim.get("include_cmb", True)
        if self.include_cmb:
            self.out_cmb_map: Asset = self.assets_out['cmb_map']
        self.out_sky_maps: Asset = self.assets_out['sky_no_noise_maps']
        # self.out_noise_maps: Asset = self.assets_out['noise_maps']
        out_cmb_map_handler: HealpyMap
        out_obs_maps_handler: HealpyMap

        # self.in_noise_cache: Asset = self.assets_in['scale_cache']
        if self.include_cmb:
            self.in_cmb_ps: AssetWithPathAlts = self.assets_in['cmb_ps']
        self.in_fg_config: Asset = self.assets_in['fg_config']
        self.in_fg_cache: Asset = self.assets_in['fg_maps']
        in_noise_cache_handler: Union[HealpyMap, NumpyPowerSpectrum]
        in_cmb_ps_handler: CambPowerSpectrum

        # Initialize constants from configs
        self.nside_sky = self.get_nside_sky()
        logger.info(f"Simulations will generated at nside_sky = {self.nside_sky}.")
        self.nside_out = cfg.scenario.nside
        logger.info(f"Simulations will be output at nside_out = {self.nside_out}")
        self.output_units = cfg.scenario.units
        self.sky_flex_unit = u.Unit(cfg.model.sim.sky_unit)
        logger.info(f"Output units are {self.output_units}")

        self.component_config = OmegaConf.to_container(cfg.model.sim.fgs, resolve=True)
        logger.info(f"Component configs are {list(dict(cfg.model.sim.fgs).keys())}")
        cfg_preset_strings = cfg.model.sim.get("preset_strings", None)
        if cfg_preset_strings is not None:
            self.preset_strings = OmegaConf.to_container(cfg_preset_strings, resolve=True)
        else:
            self.preset_strings = None
        logger.info(f"Preset strings are {self.preset_strings}")

        self.instrument: Instrument = make_instrument(cfg=cfg)
        self.do_bandpass_integration_each_sim = cfg.model.sim.do_bandpass_int_each_sim

        self.cmb_seed_factory = SeedFactory(cfg.model.sim.cmb.seed_template_map)
        self.cmb_factory = CMBFactory(cfg)

        self.cmb_beam = cfg.scenario.cmb_beam  # 0: do not apply beam to cmb; 
                                               # "min": apply lowest FWHM beam to cmb; 
                                               # other float: beam in arcmin to apply to cmb

        self.use_constant_fg = cfg.model.sim.get("use_constant_fg", None)
        self.downgrade_lmax = cfg.model.sim.downgrade_lmax

        # Do not create the Sky object here, it takes too long and will slow down initial checks
        self.sky_flex = None
        self.sky_cmb = None  # Only for use with split.fgs_fixed
        # Do not load maps until execute()
        self.fgs_const_maps = {}
        self.fgs_fixed_maps = {}  # for use if a split needs these

    def execute(self) -> None:
        """
        Creates simulations for all sims within all splits.

        Sets up the Sky object just once
           - Make placeholder object for CMB (others could be added here)
           - Preset strings are passed here
        Thus, components from preset strings are created here, once, for all simulations
        """
        logger.debug(f"Running {self.__class__.__name__} execute() method")
        if self.include_cmb:
            placeholder = [pysm3.Model(nside=self.nside_sky, max_nside=self.nside_sky)]
            placeholder_label = ['cmb']
        else:
            placeholder = None
            placeholder_label = None
        
        # Remove "dist" key from foreground configurations 
        #   (including hypothetical 2-level fgs like a1 and d4)
        for comp_dict in self.component_config.values():
            if "dist" in comp_dict:
                del comp_dict["dist"]
            for v in comp_dict.values():
                if isinstance(v, dict) and "dist" in v:
                    del v["dist"]

        if self.use_constant_fg:
            preset_strings = None
            pysm_out_unit = self.sky_flex_unit
            self.init_const_fg_maps()
            logger.info("Using fixed foreground maps instead of preset strings per sim.")
        else:
            preset_strings = self.preset_strings
            pysm_out_unit = self.output_units

        logger.debug('Creating Flexible Sky object')
        self.sky_flex = FlexSky(nside=self.nside_sky,
                                # CMB is the only placeholder used
                                component_objects=placeholder,
                                component_object_names=placeholder_label,
                                # Create the to-be-changed (d11, s6) 
                                #    components via component_config
                                component_config=self.component_config,
                                # Create static fgs via preset_strings
                                #    (only if not using constant fgs)
                                preset_strings=preset_strings,
                                output_unit=pysm_out_unit)
        logger.debug('Done creating Flexible Sky object')
        self.default_execute()
        self.purge_const_fg_maps()

    def init_const_fg_maps(self):
        # Constant foregrounds are the foregrounds constant across
        #   all splits
        if self.use_constant_fg:
            for det in self.instrument.dets.keys():
                with self.name_tracker.set_context("freq", det):
                    # use_alt_path is false for the constant fgs
                    self.fgs_const_maps[det] = self.in_fg_cache.read(use_alt_path=False)

    def purge_const_fg_maps(self):
        self.fgs_const_maps = {}

    def init_fgs_fixed(self, split: Split):
        # Fixed foregrounds are the foregrounds constant for one split
        if not split.fgs_fixed:
            return
        for det in self.instrument.dets.keys():
            with self.name_tracker.set_contexts(dict(freq=det,
                                                     split=split.name)):
                # use_alt_path is true for the fixed fgs
                self.fgs_fixed_maps[det] = self.in_fg_cache.read(use_alt_path=True)
        
        # Set up Sky object for this split
        if self.include_cmb:
            placeholder = [pysm3.Model(nside=self.nside_sky, max_nside=self.nside_sky)]
            placeholder_label = ['cmb']
        else:
            # I don't know why this would be done, but sure, I'll keep it.
            placeholder = None
            placeholder_label = None

        if self.use_constant_fg:
            preset_strings = None
        else:
            preset_strings = self.preset_strings

        logger.debug('Creating Flexible Sky object')
        self.sky_cmb = FlexSky(nside=self.nside_sky,
                               component_objects=placeholder,
                               component_object_names=placeholder_label,
                               preset_strings=preset_strings,
                               output_unit=self.sky_flex_unit)

    def purge_fgs_fixed(self):
        self.fgs_fixed_maps = {}

    def process_split(self, split: Split) -> None:
        """
        Processes all sims for a split, making simulations.
        Hollow boilerplate.

        Args:
            split (Split): The split to process.
        """
        self.init_fgs_fixed(split)
        with tqdm(total=split.n_sims, desc=f"{split.name}: ", leave=False) as pbar:
            for sim in split.iter_sims():
                pbar.set_description(f"{split.name}: {sim:04d}")
                with self.name_tracker.set_context("sim_num", sim):
                    self.process_sim(split, sim_num=sim)
                pbar.update(1)
        self.purge_fgs_fixed()

    def process_sim(self, split: Split, sim_num: int) -> None:
        """
        Produces a single simulation. Expects the sky object to be initialized.

        Args:
            split (Split): The split to process. Needed for some configuration information.
            sim_num (int): The simulation number.
        """
        this_sky = self.sky_cmb if split.fgs_fixed else self.sky_flex
        sim_name = self.name_tracker.sim_name()  # For logging and seed generation
        logger.debug(f"Creating simulation {split.name}:{sim_name}")

        cmb = None
        if self.include_cmb:
            cmb_seed = self.cmb_seed_factory.get_seed(split=split,
                                                      sim=sim_name)
            ps_path = self.in_cmb_ps.path_alt if split.ps_fidu_fixed else self.in_cmb_ps.path
            cmb = self.cmb_factory.make_cmb(cmb_seed, ps_path)
            # Replace placeholder CMB (or previous simulation's CMB) with new CMB
            this_sky.replace_component('cmb', cmb)

        # Get updated foreground parameters
        if not split.fgs_fixed:
            all_fg_params = self.in_fg_config.read(use_alt_path=False)
            for fg, fg_params in all_fg_params.items():
                is_seed = "seeds" in fg_params.keys()
                if is_seed:  # Only applies to *Realization components
                    seeds = fg_params["seeds"]["value"]
                    this_sky.redraw_component(fg, seeds)
                else:
                    this_sky.update_component(fg, fg_params)

        # Track minimum FWHM; this may be used for the CMB map
        min_fwhm = 21600 * u.arcmin  # Number of arcmin in full 360 degrees. Maybe could have used np.inf

        for freq, detector in self.instrument.dets.items():
            min_fwhm = min(min_fwhm, detector.fwhm)
            if self.instrument.bandpass_integration and self.do_bandpass_integration_each_sim:
                skymaps = this_sky.get_emission(detector.wn, detector.tx)
            else:
                skymaps = this_sky.get_emission(detector.cen_freq)

            if self.use_constant_fg:
                skymaps += self.fgs_const_maps[freq]
            if split.fgs_fixed:
                skymaps += self.fgs_fixed_maps[freq]

            n_fields_sky = skymaps.shape[0]
            n_fields_det = len(detector.fields)
            if n_fields_sky == n_fields_det:
                pass
            elif n_fields_sky == 3 and n_fields_det == 1:
                # PySM3 components always include T, Q, U; extract the temperature map
                skymaps = skymaps[0]
            # else:  # There may be other cases, but none come to mind.
            #     pass

            eq = u.cmb_equivalencies(detector.cen_freq)
            skymaps = skymaps.to(self.output_units, equivalencies=eq)

            # Use pysm3.apply_smoothing... to convolve the map with the planck detector beam
            map_smoothed = pysm3.apply_smoothing_and_coord_transform(skymaps,
                                                                     detector.fwhm,
                                                                     lmax=self.downgrade_lmax,
                                                                     output_nside=self.nside_out)
            final_map = map_smoothed  # + noise_map

            column_names = []
            for field_str in detector.fields:
                column_names.append(field_str + "_STOKES")
            with self.name_tracker.set_contexts(dict(freq=freq)):
                self.out_sky_maps.write(data=final_map, column_names=column_names)
                # if self.save_noise:
                #     self.out_noise_maps.write(data=noise_map, column_names=column_names)
            logger.debug(f"For {split.name}:{sim_name}, {freq} GHz: done with channel")
            if sim_num == 0:
                logger.info(f"For {split.name}:{sim_name}, {freq} GHz: done with channel. Beam: {detector.fwhm}")

        if self.include_cmb:
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

        if self.cmb_beam == "min":
            use_fwhm = min_fwhm
        elif self.cmb_beam is None:
            use_fwhm = 0 * u.arcmin
        else:
            use_fwhm = self.cmb_beam

        scaled_map = pysm3.apply_smoothing_and_coord_transform(cmb_realization,
                                                               fwhm=use_fwhm,
                                                               lmax=self.downgrade_lmax,
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
