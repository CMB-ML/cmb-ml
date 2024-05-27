from typing import Union
import logging

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import json

from omegaconf import DictConfig


from src.core import (
    BaseStageExecutor, 
    Split,
    Asset, AssetWithPathAlts
    )
from tqdm import tqdm

from src.core.asset_handlers.asset_handlers_base import EmptyHandler, Config # Import for typing hint
from src.core.asset_handlers.psmaker_handler import NumpyPowerSpectrum


logger = logging.getLogger(__name__)


class MakeTheoryPSStats(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="ps_theory_stats")

        self.out_wmap_ave_ps: Asset = self.assets_out["wmap_ave"]
        self.out_wmap_std_ps: Asset = self.assets_out["wmap_std"]
        out_ps_handler: NumpyPowerSpectrum

        self.in_ps_theory: AssetWithPathAlts = self.assets_in["theory_ps"]
        in_ps_handler: NumpyPowerSpectrum

        self.n_ps = self.get_stage_element("wmap_n_ps")

    def execute(self) -> None:
        # Remove this function
        logger.debug(f"Running {self.__class__.__name__} execute()")
        ave_ps, std_ps = self.estimate_wmap_ps_dist(self.n_ps)
        self.out_wmap_ave_ps.write(data=ave_ps)
        self.out_wmap_std_ps.write(data=std_ps)

    def estimate_wmap_ps_dist(self, n_ps):
        ps_idx = np.arange(0, n_ps)

        ps_samples = None

        for i, idx in enumerate(ps_idx):
            with self.name_tracker.set_contexts({"split": "Train", "sim_num": idx}):
                ps_theory = self.in_ps_theory.read(use_alt_path=False)
            if ps_samples is None:
                ps_samples = np.zeros((n_ps, ps_theory.shape[0]))
            ps_samples[i] = ps_theory

        mean_ps = np.mean(ps_samples, axis=0)
        std_ps = np.std(ps_samples, axis=0)

        return mean_ps, std_ps
