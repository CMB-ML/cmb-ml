from typing import Dict, List
from pathlib import Path

import healpy as hp

from hydra_filesets import PlanckInstrumentFiles
from planck_instrument_noise_maker import NoiseCacheCreator, random_noise_map_maker


class PlanckInstrument:
    def __init__(self, detectors) -> None:
        self._detectors: Dict[int, Detector] = detectors

    def get_noise(self, freq: int, field: str, seed):
        self.assume_freq_exists(freq)
        return self._detectors[freq].get_noise_map(field, seed)
    
    def get_beam(self, freq):
        self.assume_freq_exists(freq)
        return self._detectors[freq]

    def assume_freq_exists(self, freq):
        try:
            assert self._detectors[freq].nom_freq == freq
        except AssertionError:
            raise KeyError(f"You set this up wrong Jim")


class Detector:
    def __init__(self, nom_freq: int, cen_freq, fwhm, src_name_getter, cache_name_getter, nside):
        self.nom_freq = nom_freq
        self.cen_freq = cen_freq
        self.fwhm = fwhm
        self.cache_name_getter = cache_name_getter
        self.cache_creator = NoiseCacheCreator(src_name_getter(nom_freq),
                                               nside,
                                               self.cen_freq)

    def get_noise_map(self, field, seed):
        sd_map =  self._get_or_make_sd_map(field)
        try:
            noise_map = self.make_noise_map(sd_map, seed)
        except NotImplementedError as e:
            raise e
        return noise_map

    def _get_or_make_sd_map(self, field):
        cache_path: Path = self.cache_name_getter(self.nom_freq, field)
        if not cache_path.exists():
            self.cache_creator.create_noise_cache(field, cache_path)
        return hp.read_map(cache_path)
    
    def make_noise_map(self, sd_map, random_seed):
        return random_noise_map_maker(sd_map, random_seed, center_frequency=self.cen_freq)


def make_planck_instrument(conf):
    planck_instr_fs = PlanckInstrumentFiles(conf)
    freqs: List(int) = conf.simulation.detector_freqs
    nside: int = conf.simulation.nside

    table = planck_instr_fs.read_instr_table()

    detectors: Dict[int, Detector] = {}
    for band in freqs:
        band_str = str(band)

        try:
            assert band_str in table["band"]
        except AssertionError:
            raise KeyError(f"A detector specified in the configs, {band} " \
                            f"(converted to {band_str}) does not exist in " \
                            f"the QTable ({planck_instr_fs.table}).")

        center_frequency = table.loc[band_str]["center_frequency"]
        fwhm = table.loc[band_str]["fwhm"]
        detectors[band] = Detector(nom_freq=band,
                                    cen_freq=center_frequency,
                                    fwhm=fwhm,
                                    src_name_getter=planck_instr_fs.src.get_path_for,
                                    cache_name_getter=planck_instr_fs.cache.get_path_for,
                                    nside=nside)
    return PlanckInstrument(detectors=detectors)

def make_noise_component():
    pass