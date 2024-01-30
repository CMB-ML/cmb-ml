from typing import *
import logging
from dataclasses import dataclass, field

import hydra
from hydra.core.config_store import ConfigStore

import pysm3

from utils.hydra_log_helper import *
from hydra_filesets import DatasetFiles
from component_seed_maker import SimLevelSeedMaker, FieldLevelSeedMaker
from planck_instrument import (
    InstrumentNoise, 
    InstrumentNoiseMaker, 
    PlanckInstrument, 
    make_noise_maker, 
    make_planck_instrument )
from component_cmb import CMBMaker, make_cmb_maker

# Goal: Use a conf to make the CMB component


logger = logging.getLogger(__name__)


@dataclass
class SplitsDummy:
    ps_fidu_fixed: bool = False
    n_sims: int = 4

@dataclass
class DummyConfig:
    defaults: List[Any] = field(default_factory=lambda: [
        "config_no_out",
        "_self_"
        ])
    splits: Dict[str, SplitsDummy] = field(default_factory=lambda: {
        "Dummy0": SplitsDummy,
        "Dummy1": SplitsDummy(ps_fidu_fixed=True)
    })
    dataset_name: str = "Dummy"

cs = ConfigStore.instance()
cs.store(name="this_config", node=DummyConfig)


@hydra.main(version_base=None, config_path="cfg", config_name="this_config")
def try_cmb_from_conf(cfg):
    logger.debug(f"Running {__name__} in {__file__}")
    
    # Making global setup - the only place I want to pull from the conf
    dataset_files = DatasetFiles(cfg)
    cmb_seed_maker = SimLevelSeedMaker(cfg, "cmb")
    noise_seed_maker = FieldLevelSeedMaker(cfg, "noise")

    nside = cfg.simulation.nside
    preset_strings = ["d9"]  # Instead of getting from conf, for trial only
    planck_freqs = [100]     # Instead of getting from conf, for trial only

    planck: PlanckInstrument = make_planck_instrument(cfg)
    cmb_maker: CMBMaker = make_cmb_maker(cfg)
    noise_maker: InstrumentNoiseMaker = make_noise_maker(cfg, planck)

    # Pretend to be at sim level (no dependence on config)
    pretend_split = dataset_files.get_split("Dummy1")
    pretend_sim = pretend_split.get_sim(1)
    cmb_seed = cmb_seed_maker.get_seed(pretend_split, pretend_sim)

    cmb: pysm3.CMBLensed = cmb_maker.make_cmb_lensed(cmb_seed, pretend_sim)
    sky = pysm3.Sky(nside=nside, 
                    component_objects=[cmb],
                    preset_strings=preset_strings, 
                    output_unit="uK_RJ")
    noise: InstrumentNoise = noise_maker.make_instrument_noise()

    for nom_freq in planck_freqs:
        beam = planck.get_beam(nom_freq)
        skymaps = sky.get_emission(beam.cen_freq)
        for skymap, field_str in zip(skymaps, "TQU"):
            if nom_freq in [545, 857] and field_str != "T":
                continue
            map_smoothed = pysm3.apply_smoothing_and_coord_transform(skymap, beam.fwhm)
            noise_seed = noise_seed_maker.get_seed(pretend_split,
                                                         pretend_sim, 
                                                         field_str)
            noise_map = noise.get_noise_map(nom_freq, field_str, noise_seed)
            final_map = map_smoothed + noise_map
            logger.info(f"Success! Made map for {nom_freq} GHz.")


if __name__ == "__main__":
    try_cmb_from_conf()
