# from typing import NamedTuple

# from pathlib import Path
# from typing import List, Dict
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from omegaconf import DictConfig

from src.core import (
    BaseStageExecutor, 
    Split,
    GenericHandler,
    AssetWithPathAlts
    )
# from ..make_ps import get_power as _get_power
from src.core.asset_handlers.psmaker_handler import CambPowerSpectrum, NumpyPowerSpectrum


logger = logging.getLogger(__name__)


# class FrozenAsset(NamedTuple):
#     path: Path
#     handler: GenericHandler


# class TaskTarget(NamedTuple):
#     asset_in: FrozenAsset
#     asset_out: FrozenAsset
#     all_map_fields: str
#     detector_fields: str
#     norm_factors: float



class SerialConvertTheoryPowerSpectrumExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="theory_ps")

        self.out_theory_ps: AssetWithPathAlts = self.assets_out["theory_ps"]
        out_theory_ps_handler: NumpyPowerSpectrum

        self.in_theory_ps: AssetWithPathAlts = self.assets_in["theory_ps"]
        in_theory_ps_handler: CambPowerSpectrum

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute().")
        self.default_execute()

    def process_split(self, 
                      split: Split) -> None:
        logger.info(f"Running {self.__class__.__name__} process_split() for split: {split.name}.")
        if split.ps_fidu_fixed:
            ps = self.in_theory_ps.read(use_alt_path=True)
            self.out_theory_ps.write(use_alt_path=True, data=ps)
        else:
            for sim in tqdm(split.iter_sims()):
                with self.name_tracker.set_context("sim_num", sim):
                    ps = self.in_theory_ps.read(use_alt_path=False)
                    self.out_theory_ps.write(use_alt_path=False, data=ps)
