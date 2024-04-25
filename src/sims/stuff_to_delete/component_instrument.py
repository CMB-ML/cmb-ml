# from typing import Dict, List

# from namer_instrument import InstrumentFilesNamer


# class Instrument:
#     def __init__(self, detectors) -> None:
#         self.detectors: Dict[int, Detector] = detectors
    
#     def get_beam(self, freq):
#         return self.detectors[freq]
    
#     def iter_detectors(self):
#         for detector in self.detectors:
#             yield self.detectors[detector]


# class Detector:
#     def __init__(self, nom_freq: int, cen_freq, fwhm):
#         self.nom_freq = nom_freq
#         self.cen_freq = cen_freq
#         self.fwhm = fwhm


# def make_planck_instrument(conf):
#     planck_instr_fs = InstrumentFilesNamer(conf)
#     freqs: List[int] = conf.simulation.detector_freqs

#     table = planck_instr_fs.read_instr_table()

#     detectors: Dict[int, Detector] = {}
#     for band in freqs:
#         band_str = str(band)

#         try:
#             assert band_str in table["band"]
#         except AssertionError:
#             raise KeyError(f"A detector specified in the configs, {band} " \
#                             f"(converted to {band_str}) does not exist in " \
#                             f"the QTable ({planck_instr_fs.table}).")

#         center_frequency = table.loc[band_str]["center_frequency"]
#         fwhm = table.loc[band_str]["fwhm"]
#         detectors[band] = Detector(nom_freq=band,
#                                    cen_freq=center_frequency,
#                                    fwhm=fwhm)
#     return Instrument(detectors=detectors)
