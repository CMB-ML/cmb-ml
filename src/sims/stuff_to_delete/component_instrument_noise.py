# from typing import Dict, Callable
# from pathlib import Path
# import logging

# import healpy as hp

# import utils.fits_inspection as fits_inspect
# from component_instrument import Detector, Instrument, InstrumentFilesNamer
# from physics_instrument_noise import planck_result_to_sd_map, make_random_noise_map


# logger = logging.getLogger(__name__)


# class NoiseCacheCreator:
#     def __init__(self, source_path: Path, nside: int, center_frequency):
#         self.hdu: int = fits_inspect.ASSUME_FITS_HEADER
#         self.ref_path = source_path
#         self.nside = nside
#         self.center_frequency = center_frequency

#     def create_noise_cache(self, field_str, cache_path) -> None:
#         logger.debug(f"component_instrument_noise.create_noise_cache start")
#         field_idx = fits_inspect.lookup_field_idx(field_str, self.ref_path, self.hdu)
        
#         st_dev_skymap = planck_result_to_sd_map(fits_fn=self.ref_path, 
#                                                 hdu=self.hdu,
#                                                 field_idx=field_idx, 
#                                                 nside_out=self.nside,
#                                                 cen_freq=self.center_frequency)

#         col_names = {"T": "II", "Q": "QQ", "U": "UU"}
#         hp.write_map(filename=str(cache_path),
#                      m=st_dev_skymap,
#                      nest=False,
#                      column_names=[col_names[field_str]],
#                      column_units=["K_CMB"],
#                      dtype=st_dev_skymap.dtype,
#                      overwrite=True
#                     # TODO: figure out how to add a comment to hp's map... or switch with astropy equivalent 
#                     #  extra_header=f"Variance map pulled from {self.ref_map_fn}, {col_names[field_str]}"
#                      )
#         logger.debug(f"component_instrument_noise.create_noise_cache end")
#         return st_dev_skymap


# class InstrumentNoise:
#     def __init__(self, detector_noises) -> None:
#         self._detector_noises: Dict[int, DetectorNoise] = detector_noises
    
#     def get_noise_map(self, freq: int, field: str, seed):
#         return self._detector_noises[freq].get_noise_map(field, seed)


# class DetectorNoise:
#     def __init__(self, 
#                  detector:Detector, 
#                  src_name_getter: Callable, 
#                  cache_name_getter: Callable, 
#                  nside: int, 
#                  force_new_cache: bool):
#         self.detector = detector
#         self.cache_name_getter = cache_name_getter
#         self.cache_creator = NoiseCacheCreator(src_name_getter(self.detector.nom_freq),
#                                                nside,
#                                                self.detector.cen_freq)
#         self.force_new_cache = force_new_cache
#         self.new_cache_made = {}

#     def get_noise_map(self, field, seed):
#         sd_map =  self._get_or_make_sd_map(field)
#         noise_map = self.make_noise_map(sd_map, seed)
#         return noise_map

#     def _get_or_make_sd_map(self, field):
#         cache_path: Path = self.cache_name_getter(self.detector.nom_freq, field)
#         made: bool = self.was_cache_already_made(field)
#         exists: bool = cache_path.exists()
#         force: bool = self.force_new_cache
#         need: bool

#         # if we've marked the noise as made and it was deleted, we have an issue
#         if made and not exists:
#             raise FileNotFoundError(f"Cached noise at {cache_path} reportedly made but does not currently exist.")
        
#         # If the noise cache hasn't been made, and doesn't exist, we need to make it
#         #    if we're forcing the creation of the cache, and we didn't make it 
#         #    (for this run of simulations), we need to make it.
#         # made  force  exists  |  need
#         #  F      F      F     |    T 
#         #  F      F      T     |    F
#         #  F      T      F     |    T
#         #  F      T      T     |    T
#         #  T      F      F     |    F
#         #  T      F      T     |    F
#         #  T      T      F     |    F
#         #  T      T      T     |    F

#         need = False
#         if not made and (force or not exists):
#             need = True

#         if need:
#             self.cache_creator.create_noise_cache(field, cache_path)
#             self.mark_cache_made(field)
#         return hp.read_map(cache_path)
    
#     def was_cache_already_made(self, field) -> bool:
#         return self.new_cache_made.get(field, False)

#     def mark_cache_made(self, field) -> None:
#         self.new_cache_made[field] = True

#     def make_noise_map(self, sd_map, random_seed):
#         return make_random_noise_map(sd_map, 
#                                       random_seed, 
#                                       center_frequency=self.detector.cen_freq)


# class InstrumentNoiseFactory:
#     def __init__(self, 
#                  conf,
#                  planck: Instrument) -> None:
#         planck_instr_fs = InstrumentFilesNamer(conf)
#         self.instrument: Instrument = planck
#         self.src_name_getter=planck_instr_fs.src.get_path_for
#         self.cache_name_getter=planck_instr_fs.cache.get_path_for
#         self.nside = conf.simulation.nside_out
#         self.force_new_cache = conf.force_new_noise_cache

#     def make_instrument_noise(self):
#         detector_noises = {}
#         for detector in self.instrument.iter_detectors():
#             freq = detector.nom_freq
#             detector_noises[freq] = DetectorNoise(detector,
#                                                   src_name_getter=self.src_name_getter,
#                                                   cache_name_getter=self.cache_name_getter,
#                                                   nside=self.nside,
#                                                   force_new_cache=self.force_new_cache)
#         instrument_noise = InstrumentNoise(detector_noises)
#         return instrument_noise


# def make_noise_maker(conf, instrument):
#     noise_maker = InstrumentNoiseFactory(conf, instrument)
#     return noise_maker
