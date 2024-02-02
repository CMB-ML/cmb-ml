from typing import Dict, List
from pathlib import Path

import healpy as hp

from hydra_filesets import PlanckInstrumentFiles
from planck_instrument_noise_maker import NoiseCacheCreator, random_noise_map_maker


class PlanckInstrument:
    def __init__(self, detectors) -> None:
        self.detectors: Dict[int, Detector] = detectors
    
    def get_beam(self, freq):
        return self.detectors[freq]
    
    def iter_detectors(self):
        for detector in self.detectors:
            yield self.detectors[detector]


class Detector:
    def __init__(self, nom_freq: int, cen_freq, fwhm):
        self.nom_freq = nom_freq
        self.cen_freq = cen_freq
        self.fwhm = fwhm


def make_planck_instrument(conf):
    planck_instr_fs = PlanckInstrumentFiles(conf)
    freqs: List(int) = conf.simulation.detector_freqs

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
                                    fwhm=fwhm)
    return PlanckInstrument(detectors=detectors)


class InstrumentNoise:
    def __init__(self, detector_noises) -> None:
        self._detector_noises: Dict[int, DetectorNoise] = detector_noises
    
    def get_noise_map(self, freq: int, field: str, seed):
        return self._detector_noises[freq].get_noise_map(field, seed)


class DetectorNoise:
    def __init__(self, detector:Detector, src_name_getter, cache_name_getter, nside):
        self.detector = detector
        self.cache_name_getter = cache_name_getter
        self.cache_creator = NoiseCacheCreator(src_name_getter(self.detector.nom_freq),
                                               nside,
                                               self.detector.cen_freq)

    def get_noise_map(self, field, seed):
        sd_map =  self._get_or_make_sd_map(field)
        try:
            noise_map = self.make_noise_map(sd_map, seed)
        except NotImplementedError as e:
            raise e
        return noise_map

    def _get_or_make_sd_map(self, field):
        cache_path: Path = self.cache_name_getter(self.detector.nom_freq, field)
        if not cache_path.exists():
            self.cache_creator.create_noise_cache(field, cache_path)
        return hp.read_map(cache_path)
    
    def make_noise_map(self, sd_map, random_seed):
        return random_noise_map_maker(sd_map, 
                                      random_seed, 
                                      center_frequency=self.detector.cen_freq)


class InstrumentNoiseMaker:
    def __init__(self, 
                 conf,
                 planck: PlanckInstrument) -> None:
        planck_instr_fs = PlanckInstrumentFiles(conf)
        self.instrument: PlanckInstrument = planck
        self.src_name_getter=planck_instr_fs.src.get_path_for
        self.cache_name_getter=planck_instr_fs.cache.get_path_for
        self.nside = conf.simulation.nside

    def make_instrument_noise(self):
        detector_noises = {}
        for detector in self.instrument.iter_detectors():
            freq = detector.nom_freq
            detector_noises[freq] = DetectorNoise(detector,
                                                  src_name_getter=self.src_name_getter,
                                                  cache_name_getter=self.cache_name_getter,
                                                  nside=self.nside)
        instrument_noise = InstrumentNoise(detector_noises)
        return instrument_noise


def make_noise_maker(conf, instrument):
    noise_maker = InstrumentNoiseMaker(conf, instrument)
    return noise_maker
