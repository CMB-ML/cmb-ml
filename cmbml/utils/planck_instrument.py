from typing import Union, Iterable, Optional, List, Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf.errors import InterpolationKeyError
import pysm3.units as u

from cmbml.core.config_helper import ConfigHelper
from cmbml.core.namers import Namer
from cmbml.core.asset_handlers import QTableHandler
from cmbml.core.asset_handlers import RIMO
from cmbml.utils.physics_rimo import reduce_rimo, get_eff_cen_freq


def unpack_det_table(det_table):
    nom_freqs = list(det_table["band"])
    cen_freqs = list(det_table["center_frequency"])
    fwhms = list(det_table["fwhm"])
    table = {int(n): {"cen_freq":c, "fwhm":f} for n,c,f in zip(nom_freqs, cen_freqs, fwhms)}
    return table


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
class Detector:
    nom_freq: int
    fields: str
    cen_freq: Optional[float]
    fwhm: Optional[float]
    wn: Optional[np.ndarray]
    tx: Optional[np.ndarray]


@dataclass(frozen=True)
class Instrument:
    dets: DetsDict[int, Detector]
    map_fields: str
    bandpass_integration: bool


@dataclass(frozen=True)
class InstrumentCfg:
    # Detector-level
    nom_freqs: Iterable[int]
    det_map_fields: Iterable[str]
    fwhms: Iterable[float]
    cen_freqs: Iterable[float]
    wns: Optional[Iterable[np.ndarray]] = None
    txs: Optional[Iterable[np.ndarray]] = None
    # Instrument-level
    map_fields: str = "I"

    # Bandpass integration is used when wns and txs are present (if RIMO is used)
    bandpass_integration: bool = False

    def __post_init__(self):
        auto_flag = self.wns is not None and self.txs is not None
        object.__setattr__(self, "bandpass_integration", auto_flag)


def _build_from_instr_cfg(ic: InstrumentCfg):
    wns = ic.wns if ic.bandpass_integration else [None] * len(ic.nom_freqs)
    txs = ic.txs if ic.bandpass_integration else [None] * len(ic.nom_freqs)

    dets = {
        f: Detector(
            nom_freq=f,
            fields=ic.det_map_fields[i],
            cen_freq=ic.cen_freqs[i],
            fwhm=ic.fwhms[i],
            wn=wns[i],
            tx=txs[i],
        )
        for i, f in enumerate(ic.nom_freqs)
    }

    return Instrument(
        dets=DetsDict(dets),
        map_fields=ic.map_fields,
        bandpass_integration=ic.bandpass_integration,
    )


def make_instrument(
        cfg,
        det_info_override: Optional[Any] = None,
        min_fwhm_override: Optional[float] = None,
        use_min_fwhm: bool = True,
        use_rimo_override: bool = None
        ):
    h = ConfigHelper(cfg, stage_str='raw_in')
    n = Namer(cfg)
    assets_in = h.get_assets_in(n, stage_str='raw_in')
    lfi_asset = assets_in['lfi_rimo']
    hfi_asset = assets_in['hfi_rimo']
    deltabandpass_asset = assets_in['deltabandpass']

    use_rimo = use_rimo_override if use_rimo_override is not None else cfg.scenario.use_rimo

    nom_freqs = cfg.scenario.detector_freqs
    map_fields = cfg.scenario.map_fields

    cfg_full_instr = cfg.scenario.ref_data_release

    trim_to_mass = cfg.scenario.rimo_trim_to_mass
    compress_to_n = cfg.scenario.rimo_compress_to_n
    compress_alpha = cfg.scenario.rimo_compress_alpha

    full_instrument = {
        int(freq): v["fields"]
        for freq, v in cfg_full_instr.items()
        if freq.isdigit()
    }
    try:
        scen_fields = [full_instrument[f] for f in nom_freqs]
    except KeyError as e:
        missing = [f for f in nom_freqs if f not in full_instrument]
        raise KeyError(f"Missing detectors in ref_data_release: {missing}")
    scen_fields = [map_fields if len(map_fields) < len(s) else s for s in scen_fields]

    if det_info_override is None:
        det_info = deltabandpass_asset.read()
        det_info = unpack_det_table(det_info)
    else:
        det_info = unpack_det_table(det_info_override)

    # Determine minimum beam (FWHM) constraint
    if not use_min_fwhm:
        # Explicitly disable any lower bound on beam size
        min_obs_beam = 0.0
    else:
        try:
            # Use override if provided, else fall back to config value
            min_obs_beam = (
                cfg.model.sim.get("min_obs_beam", 0)
                if min_fwhm_override is None
                else min_fwhm_override
            )
        except (KeyError, AttributeError, InterpolationKeyError):
            min_obs_beam = 0
        # Normalize None/null values to 0
        min_obs_beam = 0 if min_obs_beam is None else float(min_obs_beam)
    # Apply the minimum FWHM constraint
    min_obs_beam = u.Quantity(min_obs_beam, u.arcmin)
    fwhms = [max(det_info[f]["fwhm"], min_obs_beam) for f in nom_freqs]

    if use_rimo:
        # If using the RIMO, then bandpass integration is used.
        # the txs and wns (transmission levels [txs] per wavenumber [wns])
        rimo = {}
        for f in nom_freqs:
            if f in [30,44,70]:
                raw_freq, raw_tx = lfi_asset.read(nom_freq=f)
            else:
                raw_freq, raw_tx = hfi_asset.read(nom_freq=f)
            red = reduce_rimo(raw_freq, raw_tx, trim_to_mass, compress_to_n, compress_alpha)
            rimo[f] = {"freq": red[0], "tx": red[1]}
    else:
        rimo = None

    cen_freqs = []
    if cfg.scenario.use_rimo_cen_freq:
        if not use_rimo:
            raise ValueError("Cannot use RIMO for center frequencies without use_rimo.")
        cen_freqs = [get_eff_cen_freq(rimo[f]["freq"], rimo[f]["tx"]) for f in nom_freqs]
    else:
        cen_freqs = [det_info[f]["cen_freq"] for f in nom_freqs]

    instr_cfg = InstrumentCfg(
        nom_freqs=nom_freqs,
        det_map_fields=scen_fields,
        fwhms=fwhms,
        cen_freqs=cen_freqs,
        wns=None if rimo is None else [rimo[f]['freq'] for f in nom_freqs],
        txs=None if rimo is None else [rimo[f]['tx'] for f in nom_freqs],
        map_fields=map_fields
    )
    instrument = _build_from_instr_cfg(instr_cfg)
    return instrument


def make_instrument_manual(
        nom_freqs: list[int],
        deltabandpass_path: Union[str, Path],
        lfi_path: Union[str, Path]=None, 
        hfi_path: Union[str, Path]=None, 
        scen_fields: Union[dict, list]=None,
        min_obs_beam=0.0,  # Set to zero to disable
        trim_to_mass=0.999,
        compress_to_n=50,
        compress_alpha=0.5,
        use_rimo=True,
        use_rimo_cen_freq=True,
        ) -> Instrument:
    # Determine which map fields are being used
    if isinstance(scen_fields, dict):
        # Check that all nom_freqs are keys of map_fields
        nf = set(nom_freqs)
        sf = set(int(k) for k in scen_fields.keys())
        if len(nf-sf) != 0:
            raise ValueError(f"nom_freqs includes {nf-sf}, but map fields does not")
        # Reduce map_fields down to the appropriate list
        scen_fields = [scen_fields[d] for d in nom_freqs]
    elif isinstance(scen_fields, list):
        if len(nom_freqs) != len(scen_fields):
            raise ValueError(f"nom_freqs ({nom_freqs}) has a different length cf map_fields ({scen_fields})")
    elif scen_fields is None:
        # Assume temperature only
        scen_fields = ["I" for _ in nom_freqs]

    # map_fields = ""
    # for sf in scen_fields:
    #     map_fields = sf if len(sf) > len(map_fields) else map_fields
    map_fields = max(scen_fields, key=len)

    det_info = QTableHandler().read(deltabandpass_path)
    det_info = unpack_det_table(det_info)

    min_obs_beam = u.Quantity(min_obs_beam, u.arcmin)
    fwhms = [max(det_info[f]["fwhm"], min_obs_beam) for f in nom_freqs]

    rimo = None
    if use_rimo:
        rimo = {}
        for f in nom_freqs:
            asset_path = lfi_path if f in [30, 44, 70] else hfi_path
            raw_freq, raw_tx = RIMO().read(asset_path, f)
            red = reduce_rimo(raw_freq, raw_tx, trim_to_mass, compress_to_n, compress_alpha)
            rimo[f] = {"freq": red[0], "tx": red[1]}

    if use_rimo_cen_freq:
        if not use_rimo:
            raise ValueError("Cannot use RIMO for center frequencies without use_rimo.")
        cen_freqs = [get_eff_cen_freq(rimo[f]["freq"], rimo[f]["tx"]) for f in nom_freqs]
    else:
        cen_freqs = [det_info[f]["cen_freq"] for f in nom_freqs]

    instr_cfg = InstrumentCfg(
        nom_freqs=nom_freqs,
        det_map_fields=scen_fields,
        fwhms=fwhms,
        cen_freqs=cen_freqs,
        wns=[rimo[f]['freq'] for f in nom_freqs],
        txs=[rimo[f]['tx'] for f in nom_freqs],
        map_fields=map_fields
    )
    instrument = _build_from_instr_cfg(instr_cfg)
    return instrument
