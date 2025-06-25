from typing import Optional, Dict
from dataclasses import dataclass
from pathlib import Path
import logging

from omegaconf.errors import InterpolationKeyError

from pysm3.units import Unit

from cmbml.utils.physics_units import convert_field_str_to_Unit

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detector:
    nom_freq: int
    fields: str
    # unit: Unit
    cen_freq: Optional[float] = None
    fwhm: Optional[float] = None


class DetsDict(dict):
    """
    A dictionary that maps detector frequencies to Detector objects.
    Raises an error if accessed while empty.
    """

    def _raise_if_empty(self, context: str = "access"):
        if not self:
            raise ValueError(f"The instrument has no detectors. "
                             f"Attempted to {context}. Ensure the "
                              "configuration has 'scenario.detector_freqs' "
                              "(or top-level 'detectors').")

    def __getitem__(self, key):
        self._raise_if_empty("access a detector")
        return super().__getitem__(key)

    def __iter__(self):
        self._raise_if_empty("iterate over detectors")
        return super().__iter__()

    def items(self):
        self._raise_if_empty("access 'dets.items()'")
        return super().items()

    def keys(self):
        self._raise_if_empty("access 'dets.keys()'")
        return super().keys()

    def values(self):
        self._raise_if_empty("access 'dets.values()'")
        return super().values()

    def get(self, key, default=None):
        self._raise_if_empty("use 'dets.get()'")
        return super().get(key, default)

    def __contains__(self, key):
        self._raise_if_empty("check if a key is in dets")
        return super().__contains__(key)


@dataclass(frozen=True)
class Instrument:
    dets: DetsDict[int, Detector]
    map_fields: str


def make_detector(det_info, band, fields):
    band_str = str(band)
    try:
        assert band_str in det_info['band']
    except AssertionError:
        raise KeyError(f"A detector specified in the configs, {band} " \
                        f"(converted to {band_str}) does not exist in " \
                        f"the given QTable path.")

    center_frequency = det_info.loc[band_str]['center_frequency']
    fwhm = det_info.loc[band_str]['fwhm']
    # unit = "MJy/sr" if band in [545, 857] else "K_CMB"
    # unit = convert_field_str_to_Unit(unit)
    return Detector(nom_freq=band, cen_freq=center_frequency, fwhm=fwhm, fields=fields)  #, unit=unit)


# def get_detector_unit(band):
#     unit = "MJy/sr" if band in [545, 857] else "K_CMB"
#     unit = convert_field_str_to_Unit(unit)
#     return unit


def make_instrument(cfg, det_info=None):
    """
    Args:
        cfg: the hydra config object
        det_info: the planck_bandpasstable

    returns a frozen dataclass containing
            detector_freqs x map_fields
            which are a subset of the full_instrument
            and the information for each from the planck_bandpasstable
    """
    scen_fields = cfg.scenario.map_fields
    full_instrument = cfg.scenario.full_instrument
    instrument_dets = {}

    # Allow for no specification of detectors, either in top-level or in scenario
    try:
        detector_freqs = cfg.scenario.detector_freqs
    except (KeyError, AttributeError, InterpolationKeyError) as e:
        detector_freqs = []

    for freq in detector_freqs:
        available_fields = full_instrument[freq]
        selected_fields = ''.join([field for field in available_fields if field in scen_fields])
        assert len(selected_fields) > 0, f"No fields were found for {freq} detector. Available fields: {available_fields}, Scenario fields: {scen_fields}."
        if det_info:
            det = make_detector(det_info, band=freq, fields=selected_fields)
        else:
            det = Detector(nom_freq=freq, fields=selected_fields)  #, unit=get_detector_unit(freq))
        instrument_dets[freq] = det
    return Instrument(dets=DetsDict(instrument_dets), map_fields=scen_fields)
