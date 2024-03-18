from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class ExperimentParameters:
    nside: int
    detector_freqs: List[int]
    map_fields: str
    precision: str
    map_fields_tuple: Tuple[int, ...] = field(default=())

    @staticmethod
    def make_fields_tuple(map_fields_str: str) -> Tuple[int, ...]:
        if map_fields_str in ["TQU", "IQU"]:
            return (0, 1, 2)
        elif map_fields_str in ["T", "I"]:
            return (0,)
        else:
            raise KeyError("Map fields listed in experiment config do not match those known.")

    @classmethod
    def from_cfg(cls, cfg):
        exp_cfg = cfg.experiment
        map_fields_tuple = cls.make_fields_tuple(exp_cfg.map_fields)
        return cls(nside=exp_cfg.nside,
                   detector_freqs=exp_cfg.detector_freqs,
                   map_fields=exp_cfg.map_fields,
                   map_fields_tuple=map_fields_tuple,
                   precision=exp_cfg.precision)
