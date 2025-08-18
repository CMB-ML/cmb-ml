from typing import Mapping, Iterable, Union, Optional, Dict
from collections.abc import Mapping as _MappingABC
from dataclasses import dataclass
from pathlib import Path
import logging

from omegaconf.errors import InterpolationKeyError

import pysm3.units as u

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


@dataclass(frozen=True)
class InstrumentCfgLite:
    map_fields: str                              # e.g., "IQU"
    full_instrument: Dict[int, str]              # freq -> available fields string, e.g. {143: "IQU", ...}
    detector_freqs: Iterable[int]                # list/iter of frequencies to instantiate
    min_obs_beam_arcmin: float = 0.0             # numeric min beam in arcmin (0 = disabled)


def make_detector(det_info, band, fields, min_fwhm):
    band_str = str(band)
    if not band_str in det_info['band']:
        raise KeyError(f"A detector specified in the configs, {band} " \
                        f"(converted to {band_str}) does not exist in " \
                        f"the given QTable path.")

    center_frequency = det_info.loc[band_str]['center_frequency']
    fwhm = max(det_info.loc[band_str]['fwhm'], min_fwhm)
    # unit = "MJy/sr" if band in [545, 857] else "K_CMB"
    # unit = convert_field_str_to_Unit(unit)
    return Detector(nom_freq=band, cen_freq=center_frequency, fwhm=fwhm, fields=fields)


def _build_instrument_from_lite(
    lite: InstrumentCfgLite,
    det_info=None,
    *,
    use_min_fwhm: bool = True,
) -> Instrument:
    scen_fields = lite.map_fields
    full_instrument = lite.full_instrument

    min_fwhm = (lite.min_obs_beam_arcmin or 0.0)
    if not use_min_fwhm:
        min_fwhm = 0.0
    min_fwhm = (min_fwhm or 0.0) * u.arcmin  # "null" / None becomes 0

    instrument_dets = {}

    for freq in lite.detector_freqs or []:
        available_fields = full_instrument[freq]
        selected_fields = ''.join([f for f in available_fields if f in scen_fields])
        assert len(selected_fields) > 0, (
            f"No fields were found for {freq} detector. "
            f"Available fields: {available_fields}, Scenario fields: {scen_fields}."
        )
        if det_info:
            det = make_detector(det_info, band=freq, fields=selected_fields, min_fwhm=min_fwhm)
        else:
            det = Detector(nom_freq=freq, fields=selected_fields)
        instrument_dets[freq] = det

    return Instrument(dets=DetsDict(instrument_dets), map_fields=scen_fields)


def make_instrument(
    cfg,
    det_info=None,
    min_fwhm_override: Optional[float] = None,
    use_min_fwhm: bool = True,
):
    """
    Create an instrument using a config.
    """
    # Safely read from cfg; tolerate absent keys & interpolations
    scen_fields = cfg.scenario.map_fields
    full_instrument = cfg.scenario.full_instrument

    try:
        detector_freqs = cfg.scenario.detector_freqs
    except (KeyError, AttributeError, InterpolationKeyError):
        detector_freqs = []

    # Pull min beam from cfg unless override provided
    try:
        min_obs_beam = (
            cfg.model.sim.get("min_obs_beam", 0) if min_fwhm_override is None else min_fwhm_override
        )
    except (KeyError, AttributeError, InterpolationKeyError):
        min_obs_beam = 0
    # Normalize None/"null" to 0
    min_obs_beam = 0 if min_obs_beam is None else float(min_obs_beam)

    lite = InstrumentCfgLite(
        map_fields=scen_fields,
        full_instrument=full_instrument,
        detector_freqs=detector_freqs,
        min_obs_beam_arcmin=min_obs_beam,
    )
    return _build_instrument_from_lite(lite, det_info=det_info, use_min_fwhm=use_min_fwhm)


def make_instrument_simple(
    *,
    map_fields: str,
    instr_and_fields: Union[Mapping[int, str], Iterable[int]],
    min_obs_beam_arcmin: float = 0.0,
    det_info=None,
    use_min_fwhm: bool = True,
    default_fields: str = "I",  # used when given only a list of freqs
):
    """
    Hydra-free way to get an Instrument, for use with external code (e.g., notebooks).

    Examples
    --------
    1) Give explicit fields per frequency
    make_instrument_simple(
        map_fields="IQU",
        instr_and_fields={30: "I", 100: "IQU", 545: "I", 857: "I"},
        min_obs_beam_arcmin=5.0,
        det_info=planck_bandpasstable,  # QTable
    )

    2) Give only the frequencies; they all get `default_fields`
    make_instrument_simple(
        map_fields="IQU",
        instr_and_fields=[30, 100, 857],
        default_fields="I",
    )
    """
    # Normalize instr_and_fields -> Mapping[int, str]
    if isinstance(instr_and_fields, _MappingABC):
        instr_map = dict(instr_and_fields)  # copy to a plain dict
    else:
        # Avoid treating a single string like an iterable of chars
        if isinstance(instr_and_fields, (str, bytes)):
            raise TypeError(
                "instr_and_fields cannot be a string/bytes. "
                "Pass a Mapping[int, str] or an Iterable[int]."
            )
        instr_map = {int(f): default_fields for f in instr_and_fields}

    # Optional: validate fields are subsets of map_fields (helpful error early)
    allowed = set(map_fields)
    for f, fields in instr_map.items():
        if not set(fields).issubset(allowed):
            raise ValueError(
                f"Fields '{fields}' for freq {f} not subset of map_fields='{map_fields}'."
            )

    lite = InstrumentCfgLite(
        map_fields=map_fields,
        full_instrument=instr_map,                        # freq -> available fields
        detector_freqs=list(instr_map.keys()),
        min_obs_beam_arcmin=float(min_obs_beam_arcmin or 0.0),
    )
    return _build_instrument_from_lite(
        lite, det_info=det_info, use_min_fwhm=use_min_fwhm
    )


# def make_instrument(cfg, det_info=None, min_fwhm_override=None, use_min_fwhm=True):
#     """
#     Args:
#         cfg: the hydra config object
#         det_info: the planck_bandpasstable
#         min_fwhm_override: for use with hydraConfigChecker; not recommended elsewhere
#         use_min_fwhm: (bool) allows disabling min_fwhm (e.g., when downgrading using Planck instrument)
#     returns a frozen dataclass containing
#             detector_freqs x map_fields
#             which are a subset of the full_instrument
#             and the information for each from the planck_bandpasstable
#     """
#     scen_fields = cfg.scenario.map_fields
#     full_instrument = cfg.scenario.full_instrument
#     min_fwhm = cfg.model.sim.get("min_obs_beam", 0) if min_fwhm_override is None else min_fwhm_override
#     if min_fwhm is None:  # Handle user putting "null" in config yaml
#         min_fwhm = 0
#     if not use_min_fwhm:
#         min_fwhm = 0
#     min_fwhm = min_fwhm * u.arcmin

#     instrument_dets = {}

#     # Allow for no specification of detectors, either in top-level or in scenario
#     try:
#         detector_freqs = cfg.scenario.detector_freqs
#     except (KeyError, AttributeError, InterpolationKeyError) as e:
#         detector_freqs = []

#     for freq in detector_freqs:
#         available_fields = full_instrument[freq]
#         selected_fields = ''.join([field for field in available_fields if field in scen_fields])
#         assert len(selected_fields) > 0, f"No fields were found for {freq} detector. Available fields: {available_fields}, Scenario fields: {scen_fields}."
#         if det_info:
#             det = make_detector(det_info, band=freq, fields=selected_fields, min_fwhm=min_fwhm)
#         else:
#             det = Detector(nom_freq=freq, fields=selected_fields)  #, unit=get_detector_unit(freq))
#         instrument_dets[freq] = det
#     instrument = Instrument(dets=DetsDict(instrument_dets), map_fields=scen_fields)
#     return instrument
