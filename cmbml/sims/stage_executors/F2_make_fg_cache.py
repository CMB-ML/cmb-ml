import logging

from omegaconf import DictConfig, OmegaConf

import pysm3
import pysm3.units as u
from cmbml.utils.planck_instrument import make_instrument, Instrument

from cmbml.core import (
    BaseStageExecutor,
    Asset
)

from cmbml.core.asset_handlers.qtable_handler import QTableHandler # Import to register handler
from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap # Import for VS Code hints
from cmbml.utils.pysm_flex_sky import FlexSky


logger = logging.getLogger(__name__)


class PySMForegroundPrepExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="pysm_fg_prep")

        self.out_fg_map: Asset = self.assets_out['fg_maps']
        out_map_handler: HealpyMap
        self.in_fg_config: Asset = self.assets_in.get('fg_config', None)
        self.in_fg_map: Asset = self.assets_in['fg_maps']

        self.instrument: Instrument = make_instrument(cfg=cfg)

        self.nside_sky = self.get_nside_sky()
        sky_unit = cfg.model.sim.sky_unit  # Pretty sure it needs to be MJy/sr
        self.sky_unit = u.Unit(sky_unit)
        self.preset_strings = OmegaConf.to_container(cfg.model.sim.preset_strings, resolve=True)
        self.use_constant_fg = cfg.model.sim.get("use_constant_fg", None)
        self.component_config = OmegaConf.to_container(cfg.model.sim.fgs, resolve=True)
        self.do_bandpass_integration_each_sim = cfg.model.sim.do_bandpass_int_each_sim

        # A bit lazy here; I should make this two separate executors, but this way I can
        #   simply load 
        self.fixed_fg_maps = {}

    def execute(self) -> None:
        # Constant foregrounds are constant across all simulations, 
        #    regardless of split (e.g., same f1 in all sims)
        if self.use_constant_fg:
            self.make_constant_fgs()
        # Fixed foregrounds are for a particular split 
        #    (e.g., d11 and s6 in the TestFFN)
        self.make_fixed_fgs()
        self.purge_loaded_maps()

    def purge_loaded_maps(self):
        self.fixed_fg_maps = None

    def make_constant_fgs(self) -> None:
        logger.info("Making Sky for constant foreground map.")
        sky = pysm3.Sky(nside=self.nside_sky, 
                        preset_strings=self.preset_strings,
                        output_unit=self.sky_unit)
        for det in self.instrument.dets.values():
            logger.info(f"Producing constant fg map for {det.nom_freq} GHz.")
            if self.instrument.bandpass_integration:
                skymap = sky.get_emission(det.wn, det.tx)
            else:
                skymap = sky.get_emission(det.cen_freq)

            n_fields_sky = skymap.shape[0]
            n_fields_det = len(det.fields)
            if n_fields_sky == n_fields_det:
                pass
            elif n_fields_sky == 3 and n_fields_det == 1:
                # PySM3 components always include T, Q, U; extract the temperature map
                skymap = skymap[0]

            with self.name_tracker.set_context("freq", det.nom_freq):
                self.out_fg_map.write(data=skymap, use_alt_path=False)
            
            # Store maps now in case they're needed for fixed fgs
            self.fixed_fg_maps[det.nom_freq] = skymap

    def load_constant_fg_maps(self):
        for det in self.instrument.dets.values():
            with self.name_tracker.set_context("freq", det.nom_freq):
                self.fixed_fg_maps[det.nom_freq] = self.in_fg_map.read(use_alt_path=False)

    def make_fixed_fgs(self) -> None:
        # If this split needs a single set of fixed foregrounds for all simulations...
        for split in self.splits:
            if not split.fgs_fixed:
                continue

            logger.info(f"Making Sky for fixed foreground map (split: {split.name})")

            if len(self.fixed_fg_maps) == 0:
                self.load_constant_fg_maps()

            for comp_dict in self.component_config.values():
                if "dist" in comp_dict:
                    del comp_dict["dist"]
                for v in comp_dict.values():
                    if isinstance(v, dict) and "dist" in v:
                        del v["dist"]

            with self.name_tracker.set_context('split', split.name):
                try:
                    # Crash here if fg_config asset isn't set up with path_alt.
                    #   Crash later if the file itself doesn't exist. 
                    #   Hope this helps, future-me
                    self.in_fg_config.path_alt
                except AttributeError:
                    raise NotImplementedError(f"fg_config not set in pipeline")
                all_fg_params = self.in_fg_config.read(use_alt_path=True)

            # Set seeds ahead of time, set other parameters using FlexSky method
            #    Updating seeds is generates foregrounds twice and I'm lazy.
            for fg, fg_params in all_fg_params.items():
                if "seeds" in fg_params.keys():  # Only applies to *Realization components
                    seeds = fg_params["seeds"]["value"]
                    self.component_config[fg]["seeds"] = seeds

            sky = FlexSky(nside=self.nside_sky,
                          component_config=self.component_config,
                          # If we're using the constant fgs, the preset_string 
                          #    map is already generated. Do not include those.
                          preset_strings=None if self.use_constant_fg else self.preset_strings,
                          output_unit=self.sky_unit)

            # Now update the parameters. This isn't used in the current CMB-ML, but
            #   does change the SED-governing parameters as intended.
            for fg, fg_params in all_fg_params.items():
                is_seeds = "seeds" in fg_params.keys()
                if not is_seeds:
                    sky.update_component(fg, fg_params)

            # Produce maps for each frequency
            for freq, detector in self.instrument.dets.items():
                if self.instrument.bandpass_integration and self.do_bandpass_integration_each_sim:
                    skymaps = sky.get_emission(detector.wn, detector.tx)
                else:
                    skymaps = sky.get_emission(detector.cen_freq)

                # Handle polarization fields (currently, drop them)
                n_fields_sky = skymaps.shape[0]
                n_fields_det = len(detector.fields)
                if n_fields_sky == n_fields_det:
                    pass
                elif n_fields_sky == 3 and n_fields_det == 1:
                    # PySM3 components always include T, Q, U; extract the temperature map
                    skymaps = skymaps[0]
                # else:  # There may be other cases, but none come to mind.
                #     pass

                if self.use_constant_fg:
                    skymaps = skymaps + self.fixed_fg_maps[freq]

                column_names = []
                for field_str in detector.fields:
                    column_names.append(field_str + "_STOKES")
                with self.name_tracker.set_contexts(dict(freq=freq, split=split.name)):
                    self.out_fg_map.write(data=skymaps, 
                                          column_names=column_names, 
                                          # Use alternate path for the *fixed* (not constant) fgs
                                          use_alt_path=True)
                logger.debug(f"For {split.name}, {freq} GHz: done with channel for fixed fg map")

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
