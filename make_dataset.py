from typing import *
import logging

import pysm3

import hydra
from component_cmb import (CMBFactory, 
                           make_cmb_maker, 
                           save_der_cmb_ps, 
                           save_fid_cmb_map)
from component_seed_maker import (SimLevelSeedFactory,
                                  FieldLevelSeedFactory)
from component_instrument import Instrument, make_planck_instrument
from component_instrument_noise import (InstrumentNoise, 
                                        InstrumentNoiseFactory, 
                                        make_noise_maker)

from namer_dataset_output import DatasetFilesNamer
from maker_of_configs import DatasetConfigsMaker


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def make_all_simulations(cfg):
    logger.debug(f"Running {__name__} in {__file__}")
    
    logger.debug("Creating dataset directory structure")
    dataset_configs_builder = DatasetConfigsMaker(cfg)
    dataset_configs_builder.setup_folders()

    logger.debug("Creating split configurations")
    seed = cfg.simulation.cmb.wmap_indcs_seed
    chain_idcs_dict = dataset_configs_builder.make_chain_idcs_for_each_split(seed)
    dataset_configs_builder.make_split_configs(chain_idcs_dict)
    
    logger.debug("Creating cosmological parameters")
    dataset_configs_builder.make_cosmo_param_configs()

    logger.debug("Creating directing objects from source configuration files")
    dataset_files = DatasetFilesNamer(cfg)
    nside = cfg.simulation.nside
    preset_strings = list(cfg.simulation.preset_strings)
    planck_freqs = list(cfg.simulation.detector_freqs)
    field_strings = list(cfg.simulation.fields)
    lmax_pysm3_smoothing = int(cfg.simulation.cmb.derived_ps_nsmax_x * nside)
    lmax_derived_ps = int(cfg.simulation.cmb.derived_ps_nsmax_x * nside)

    planck: Instrument = make_planck_instrument(cfg)

    cmb_factory: CMBFactory = make_cmb_maker(cfg)
    cmb_seed_factory = SimLevelSeedFactory(cfg, "cmb")

    noise_factory: InstrumentNoiseFactory = make_noise_maker(cfg, planck)
    noise_seed_factory = FieldLevelSeedFactory(cfg, "noise")
    noise: InstrumentNoise = noise_factory.make_instrument_noise()

    logger.debug("Done with source configuration files")

    logger.debug("Creating datasets")
    for split in dataset_files.iter_splits():
        for sim in split.iter_sims():
            logger.debug(f"Creating simulation {split.name}:{sim.sim_num}")

            cmb_seed = cmb_seed_factory.get_seed(split, sim)
            cmb: pysm3.CMBLensed = cmb_factory.make_cmb_lensed(cmb_seed, sim)
            
            logger.debug(f"Making sky for {sim.name}")
            sky = pysm3.Sky(nside=nside, 
                            component_objects=[cmb],
                            preset_strings=preset_strings, 
                            output_unit="uK_RJ")
            logger.debug(f"Done making sky for {sim.name}")

            save_fid_cmb_map(cmb, sim)
            logger.debug(f"Writing derived ps at ell_max = {lmax_derived_ps} for {sim.name}")
            save_der_cmb_ps(cmb, sim, lmax=lmax_derived_ps)

            for nom_freq in planck_freqs:
                beam = planck.get_beam(nom_freq)
                logger.debug(f"For {sim.name}, {nom_freq} GHz: Getting sky emission.")
                skymaps = sky.get_emission(beam.cen_freq)
                logger.debug(f"For {sim.name}, {nom_freq} GHz: Getting sky emission done.")
                obs_map = []
                for skymap, field_str in zip(skymaps, field_strings):
                    if nom_freq in [545, 857] and field_str != "T":
                        logger.debug(f"For {sim.name}, {nom_freq} GHz, {field_str}: Skipping output; source data at these frequencies was not found.")
                        continue
                    logger.debug(f"For {sim.name}, {nom_freq} GHz, {field_str}: applying beam effects.")
                    map_smoothed = pysm3.apply_smoothing_and_coord_transform(skymap, beam.fwhm, lmax=lmax_pysm3_smoothing)
                    logger.debug(f"For {sim.name}, {nom_freq} GHz, {field_str}: making noise.")
                    noise_seed = noise_seed_factory.get_seed(split,
                                                        sim, 
                                                        nom_freq,
                                                        field_str)
                    noise_map = noise.get_noise_map(nom_freq, field_str, noise_seed)
                    final_map = map_smoothed + noise_map
                    obs_map.append(final_map)
                    logger.debug(f"For {sim.name}, {nom_freq} GHz, {field_str}: done with field.")
                logger.debug(f"For {sim.name}, {nom_freq} GHz: writing file.")
                sim.write_obs_map(obs_map, nom_freq)
                logger.info(f"For {sim.name}, {nom_freq} GHz: done with channel.")
            logger.debug(f"For {sim.name}: done with simulation.")


if __name__ == "__main__":
    make_all_simulations()
