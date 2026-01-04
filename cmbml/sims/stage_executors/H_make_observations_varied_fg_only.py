# Cruft. I do not recall the purpose of this precisely. 
# I think it was part of the patchwork method to get the TestFFN working, but cannot recall.


# import logging

# from omegaconf import DictConfig, OmegaConf

# import pysm3.units as u
# from tqdm import tqdm

# from cmbml.utils.planck_instrument import make_instrument, Instrument

# from cmbml.core import (
#     BaseStageExecutor,
#     Split,
#     Asset
# )

# from cmbml.core.asset_handlers.config_handler import Config
# from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap # Import for VS Code hints

# from cmbml.utils.pysm_flex_sky import FlexSky


# logger = logging.getLogger(__name__)


# class VariedFGsExecutor(BaseStageExecutor):
#     def __init__(self, cfg: DictConfig) -> None:
#         # The following stage_str must match the pipeline yaml
#         super().__init__(cfg, stage_str='make_varied_fgs')

#         self.out_sky_maps: Asset = self.assets_out['varied_fgs']
#         out_obs_maps_handler: HealpyMap

#         self.in_fg_config: Asset = self.assets_in['fg_config']
#         in_config_handler: Config

#         # Initialize constants from configs
#         self.nside_sky = self.get_nside_sky()
#         logger.info(f"Simulations will generated at nside_sky = {self.nside_sky}.")
#         self.sky_unit = u.Unit(cfg.model.sim.sky_unit)
#         logger.info(f"Sky units are {self.sky_unit}")

#         self.component_config = OmegaConf.to_container(cfg.model.sim.fgs, resolve=True)
#         logger.info(f"Component configs are {list(dict(cfg.model.sim.fgs).keys())}")
#         cfg_preset_strings = cfg.model.sim.get("preset_strings", None)
#         if cfg_preset_strings is not None:
#             self.preset_strings = OmegaConf.to_container(cfg_preset_strings, resolve=True)
#         else:
#             self.preset_strings = None
#         logger.info(f"Preset strings are {self.preset_strings}")

#         self.instrument: Instrument = make_instrument(cfg=cfg)
#         self.do_bandpass_integration_each_sim = cfg.model.sim.do_bandpass_int_each_sim
#         self.sky = None

#     def execute(self) -> None:
#         """
#         Creates simulations for all sims within all splits.

#         Sets up the Sky object just once
#            - Make placeholder object for CMB (others could be added here)
#            - Preset strings are passed here
#         Thus, components from preset strings are created here, once, for all simulations
#         """
#         logger.debug(f"Running {self.__class__.__name__} execute() method")
        
#         # Remove "dist" key from foreground configurations 
#         #   (including hypothetical 2-level fgs like a1 and d4)
#         for comp_dict in self.component_config.values():
#             if "dist" in comp_dict:
#                 del comp_dict["dist"]
#             for v in comp_dict.values():
#                 if isinstance(v, dict) and "dist" in v:
#                     del v["dist"]

#         logger.debug('Creating Flexible Sky object')
#         self.sky = FlexSky(nside=self.nside_sky,
#                            component_objects=None,
#                            component_object_names=None,
#                            component_config=self.component_config,
#                            output_unit=self.sky_unit)
#         logger.debug('Done creating Flexible Sky object')
#         self.default_execute()

#     def process_split(self, split: Split) -> None:
#         """
#         Processes all sims for a split, making simulations.
#         Hollow boilerplate.

#         Args:
#             split (Split): The split to process.
#         """
#         with tqdm(total=split.n_sims, desc=f"{split.name}: ", leave=False) as pbar:
#             for sim in split.iter_sims():
#                 pbar.set_description(f"{split.name}: {sim:04d}")
#                 with self.name_tracker.set_context("sim_num", sim):
#                     self.process_sim(split, sim_num=sim)
#                 pbar.update(1)

#     def process_sim(self, split: Split, sim_num: int) -> None:
#         """
#         Produces a single simulation. Expects the sky object to be initialized.

#         Args:
#             split (Split): The split to process. Needed for some configuration information.
#             sim_num (int): The simulation number.
#         """
#         sim_name = self.name_tracker.sim_name()  # For logging and seed generation
#         logger.debug(f"Creating simulation {split.name}:{sim_name}")

#         # Get updated foreground parameters
#         all_fg_params = self.in_fg_config.read()
#         for fg, fg_params in all_fg_params.items():
#             is_seed = "seeds" in fg_params.keys()
#             if is_seed:  # Only applies to *Realization components
#                 seeds = fg_params["seeds"]["value"]
#                 self.sky.redraw_component(fg, seeds)
#             else:
#                 self.sky.update_component(fg, fg_params)

#         for freq, detector in self.instrument.dets.items():
#             if self.instrument.bandpass_integration and self.do_bandpass_integration_each_sim:
#                 skymaps = self.sky.get_emission(detector.wn, detector.tx)
#             else:
#                 skymaps = self.sky.get_emission(detector.cen_freq)

#             n_fields_sky = skymaps.shape[0]
#             n_fields_det = len(detector.fields)
#             if n_fields_sky == n_fields_det:
#                 pass
#             elif n_fields_sky == 3 and n_fields_det == 1:
#                 # PySM3 components always include T, Q, U; extract the temperature map
#                 skymaps = skymaps[0]

#             column_names = []
#             for field_str in detector.fields:
#                 column_names.append(field_str + "_STOKES")
#             with self.name_tracker.set_contexts(dict(freq=freq)):
#                 self.out_sky_maps.write(data=skymaps, column_names=column_names)
#             logger.debug(f"For {split.name}:{sim_name}, {freq} GHz: done with channel")
#             if sim_num == 0:
#                 logger.info(f"For {split.name}:{sim_name}, {freq} GHz: done with channel. Beam: {detector.fwhm}")
#         logger.debug(f"For {split.name}:{sim_name}, done with simulation")

#     def get_nside_sky(self):
#         """
#         Returns the nside to use for PySM3's sky object. May be set with one of two 
#         configuration options.
#         """
#         nside_out = self.cfg.scenario.nside
#         nside_sky_set = self.cfg.model.sim.get("nside_sky", None)
#         nside_sky_factor = self.cfg.model.sim.get("nside_sky_factor", None)

#         nside_sky = nside_sky_set if nside_sky_set else nside_out * nside_sky_factor
#         return nside_sky
