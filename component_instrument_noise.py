from typing import Dict
from pathlib import Path
import logging

import healpy as hp

import utils.fits_inspection as fits_inspect
from component_instrument import Detector, Instrument, InstrumentFilesNamer
from physics_instrument_noise import planck_result_to_sd_map, make_random_noise_map


logger = logging.getLogger(__name__)


class NoiseCacheCreator:
    def __init__(self, source_path: Path, nside: int, center_frequency):
        self.hdu: int = fits_inspect.ASSUME_FITS_HEADER
        self.ref_path = source_path
        self.nside = nside
        self.center_frequency = center_frequency

    def create_noise_cache(self, field_str, cache_path) -> None:
        logger.debug(f"component_instrument_noise.create_noise_cache start")
        field_idx = fits_inspect.lookup_field_idx(field_str, self.ref_path, self.hdu)
        
        st_dev_skymap = planck_result_to_sd_map(fits_fn=self.ref_path, 
                                                hdu=self.hdu,
                                                field_idx=field_idx, 
                                                nside_out=self.nside,
                                                cen_freq=self.center_frequency)

        col_names = {"T": "II", "Q": "QQ", "U": "UU"}
        hp.write_map(filename=str(cache_path),
                     m=st_dev_skymap,
                     nest=False,
                     column_names=[col_names[field_str]],
                     column_units=["K_CMB"],
                     dtype=st_dev_skymap.dtype,
                     overwrite=True
                    # TODO: figure out how to add a comment to hp's map... or switch with astropy equivalent 
                    #  extra_header=f"Variance map pulled from {self.ref_map_fn}, {col_names[field_str]}"
                     )
        logger.debug(f"component_instrument_noise.create_noise_cache end")
        return st_dev_skymap


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
        return make_random_noise_map(sd_map, 
                                      random_seed, 
                                      center_frequency=self.detector.cen_freq)


class InstrumentNoiseFactory:
    def __init__(self, 
                 conf,
                 planck: Instrument) -> None:
        planck_instr_fs = InstrumentFilesNamer(conf)
        self.instrument: Instrument = planck
        self.src_name_getter=planck_instr_fs.src.get_path_for
        self.cache_name_getter=planck_instr_fs.cache.get_path_for
        self.nside = conf.simulation.nside_out

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
    noise_maker = InstrumentNoiseFactory(conf, instrument)
    return noise_maker
