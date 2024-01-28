from typing import *
import logging
from dataclasses import dataclass, field

import hydra
from hydra.core.config_store import ConfigStore

import pysm3

from utils.hydra_log_helper import *
from planck_instrument import PlanckInstrument, make_planck_instrument


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
    # simple_trial(cfg)
    nside = cfg.simulation.nside
    preset_strings = ["d9"]
    sky = pysm3.Sky(nside=nside, preset_strings=preset_strings, output_unit="uK_RJ")
    planck_freqs = [100, 217, 545, 857]
    planck:PlanckInstrument = make_planck_instrument(cfg)

    seed = 0

    for nom_freq in planck_freqs:
        beam = planck.get_beam(nom_freq)
        skymaps = sky.get_emission(beam.cen_freq)
        for skymap, field_str in zip(skymaps, "TQU"):
            if nom_freq in [545, 857] and field_str != "T":
                continue
            map_smoothed = pysm3.apply_smoothing_and_coord_transform(skymap, beam.fwhm)
            noise_map = planck.get_noise(nom_freq, field_str, seed)
            final_map = map_smoothed + noise_map
            logger.info(f"Success! Made map for {nom_freq} GHz.")


def simple_trial(cfg):
    planck = make_planck_instrument(cfg)
    planck.get_noise(30, "T", seed=0)
    planck.get_noise(30, "Q", seed=0)
    try:
        planck.get_noise(857, "Q", seed=0)
    except ValueError:
        logger.info("Correctly found that detector 857 has no polarization.")



if __name__ == "__main__":
    try_planck_instrument()
