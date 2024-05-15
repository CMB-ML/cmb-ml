from typing import Union
import logging

import numpy as np

import matplotlib.pyplot as plt
from omegaconf import DictConfig

from core import (
    BaseStageExecutor, 
    Split,
    Asset, AssetWithPathAlts
    )
from tqdm import tqdm

from core.asset_handlers.asset_handlers_base import EmptyHandler # Import for typing hint
from core.asset_handlers.psmaker_handler import NumpyPowerSpectrum


logger = logging.getLogger(__name__)


class ShowSinglePsFigExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="ps_fig")

        self.out_ps_figure: Asset = self.assets_out["ps_figure"]
        out_ps_figure_handler: EmptyHandler

        self.in_ps_theory: AssetWithPathAlts = self.assets_in["theory_ps"]
        self.in_ps_real: Asset = self.assets_in["auto_real"]
        self.in_ps_pred: Asset = self.assets_in["auto_pred"]
        out_ps_handler: NumpyPowerSpectrum

        self.fig_n_override = self.get_override_sim_nums()

    def execute(self) -> None:
        # Remove this function
        logger.debug(f"Running {self.__class__.__name__} execute()")
        for split in self.splits:
            with self.name_tracker.set_context("split", split.name):
                self.process_split(split)
            break

    def process_split(self, 
                      split: Split) -> None:
        logger.info(f"Running {self.__class__.__name__} process_split() for split: {split.name}.")

        # We may want to process a subset of all sims
        if self.fig_n_override is None:
            sim_iter = split.iter_sims()
        else:
            sim_iter = self.fig_n_override

        if split.ps_fidu_fixed:
            ps_theory = self.in_ps_theory.read(use_alt_path=True)
        else:
            ps_theory = None

        for sim in tqdm(sim_iter):
            with self.name_tracker.set_context("sim_num", sim):
                self.process_sim(ps_theory)

    def process_sim(self, ps_theory) -> None:
        for epoch in self.model_epochs:
            with self.name_tracker.set_context("epoch", epoch):
                self.process_epoch(ps_theory)

    def process_epoch(self, ps_theory) -> None:
        epoch = self.name_tracker.context['epoch']
        split = self.name_tracker.context['split']
        sim_num = self.name_tracker.context['sim_num']
        ps_real = self.in_ps_real.read()
        ps_pred = self.in_ps_pred.read()

        if ps_theory is None:
            ps_theory = self.in_ps_theory.read(use_alt_path=False)

        self.make_ps_figure(ps_real, ps_pred, ps_theory)

        plt.suptitle(f"CMBNNCS After Training for {epoch} Epochs, {split}:{sim_num}")

        self.out_ps_figure.write()
        fn = self.out_ps_figure.path
        plt.savefig(fn)
        plt.close()

    def make_ps_figure(self, ps_real, ps_pred, ps_theory):
        n_ells = ps_real.shape[0]
        ells = np.arange(1, n_ells+1)
        norm = ells * (ells + 1) / (2 * np.pi)
        plt.plot(ps_real * norm, label="Realization")
        plt.plot(ps_pred * norm, label="Prediction")
        plt.plot(ps_theory[:n_ells], label="Theory")
        plt.legend()
