from typing import *
import logging
from dataclasses import dataclass, field

import hydra
from hydra.core.config_store import ConfigStore

import pysm3

from planck_instrument import InstrumentNoise, InstrumentNoiseMaker, PlanckInstrument, make_planck_instrument


# Goal: Use the Planck Instrument


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
def try_planck_instrument(cfg):
    logger.debug(f"Running {__name__} in {__file__}")
    
    # Making global setup - the only place I want to pull from the conf
    # simple_trial(cfg)
    nside = cfg.simulation.nside
    preset_strings = ["d9"]
    planck_freqs = [100]
    
    planck:PlanckInstrument = make_planck_instrument(cfg)
    noise_maker = InstrumentNoiseMaker(cfg, planck)
    
    # Pretend to be at sim level (no dependence on config)
    seed = 0
    sky = pysm3.Sky(nside=nside, preset_strings=preset_strings, output_unit="uK_RJ")
    noise: InstrumentNoise = noise_maker.make_instrument_noise()

    for nom_freq in planck_freqs:
        beam = planck.get_beam(nom_freq)
        skymaps = sky.get_emission(beam.cen_freq)
        for skymap, field_str in zip(skymaps, "TQU"):
            if nom_freq in [545, 857] and field_str != "T":
                continue
            map_smoothed = pysm3.apply_smoothing_and_coord_transform(skymap, beam.fwhm)
            noise_map = noise.get_noise_map(nom_freq, field_str, seed)
            final_map = map_smoothed + noise_map
            logger.info(f"Success! Made map for {nom_freq} GHz.")


if __name__ == "__main__":
    try_planck_instrument()
