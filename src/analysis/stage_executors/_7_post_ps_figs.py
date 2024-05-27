from typing import Union
import logging

import numpy as np

import matplotlib.pyplot as plt
from omegaconf import DictConfig

import json
from tqdm import tqdm

from src.core import (
    BaseStageExecutor, 
    Split,
    Asset, AssetWithPathAlts
    )

from src.core.asset_handlers.asset_handlers_base import EmptyHandler # Import for typing hint
from src.core.asset_handlers.psmaker_handler import NumpyPowerSpectrum


logger = logging.getLogger(__name__)


class PostAnalysisPsFigExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="post_ps_fig")

        self.out_ps_figure_theory: Asset = self.assets_out["ps_figure_theory"]
        self.out_ps_figure_real: Asset = self.assets_out["ps_figure_real"]
        out_ps_figure_handler: EmptyHandler

        self.in_ps_theory: AssetWithPathAlts = self.assets_in["theory_ps"]
        self.in_ps_real: Asset = self.assets_in["auto_real"]
        self.in_ps_pred: Asset = self.assets_in["auto_pred"]
        self.in_wmap_distribution: Asset = self.assets_in["wmap_distribution"]
        self.in_error_distribution: Asset = self.assets_in["error_distribution"]
        in_ps_handler: NumpyPowerSpectrum

        # self.override_sim_nums = self.override_sim_nums

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
        if self.override_sim_nums is None:
            sim_iter = split.iter_sims()
        else:
            sim_iter = self.override_sim_nums

        if split.ps_fidu_fixed:
            ps_theory = self.in_ps_theory.read(use_alt_path=True)
        else:
            ps_theory = None
        
        for sim in sim_iter:
            print(f"Processing split {split.name}, sim {sim}")
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

        with open(self.in_wmap_distribution.path, 'r') as f:
            wmap_data = json.load(f)
        with open(self.in_error_distribution.path, 'r') as f:
            error_data = json.load(f)

        wmap_mean = np.array(wmap_data['wmap_mean'])
        wmap_std = np.array(wmap_data['wmap_std'])

        error_mean = np.array(error_data[f'{epoch}_mean'])
        error_std = np.array(error_data[f'{epoch}_std'])

        wmap_band = [(wmap_mean - wmap_std, wmap_mean + wmap_std), (wmap_mean - 2*wmap_std, wmap_mean + 2*wmap_std)]
        error_band = [(error_mean - error_std, error_mean + error_std), (error_mean - 2*error_std, error_mean + 2*error_std)]

        self.make_ps_figure(ps_real, ps_pred, ps_theory, wmap_band, error_band, baseline="real")
        plt.suptitle(f"CMBNNCS After Training for {epoch} Epochs, {split}:{sim_num}")
        self.out_ps_figure_real.write()
        fn = self.out_ps_figure_real.path
        print(f'writing to {fn}')
        plt.savefig(fn)
        plt.close()

        self.make_ps_figure(ps_real, ps_pred, ps_theory, wmap_band, error_band, baseline="theory")
        plt.suptitle(f"CMBNNCS After Training for {epoch} Epochs, {split}:{sim_num}")
        self.out_ps_figure_theory.write()
        fn = self.out_ps_figure_theory.path
        print(f'writing to {fn}')
        plt.savefig(fn)
        plt.close()

    # def make_ps_figure(self, ps_real, ps_pred, ps_theory):
    #     n_ells = ps_real.shape[0]
    #     ells = np.arange(1, n_ells+1)
    #     norm = ells * (ells + 1) / (2 * np.pi)
    #     plt.plot(ps_real * norm, label="Realization")
    #     plt.plot(ps_pred * norm, label="Prediction")
    #     plt.plot(ps_theory[:n_ells], label="Theory")
    #     plt.legend()

    def make_ps_figure(self, ps_real, ps_pred, ps_theory, wmap_band, error_band, baseline="real"):
        n_ells = ps_real.shape[0]
        ells = np.arange(1, n_ells+1)

        pred_conved = ps_pred
        real_conved = ps_real

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 6))

        ax1.fill_between(ells, wmap_band[0][0][:n_ells], wmap_band[0][1][:n_ells], color='limegreen', alpha=0.5, label=r'1$\sigma$ WMAP')
        ax1.fill_between(ells, wmap_band[1][0][:n_ells], wmap_band[1][1][:n_ells], color='lime', alpha=0.5, label=r'2$\sigma$ WMAP')
        ax1.plot(ells, ps_theory[:n_ells], label='Theory', color='black')
        ax1.scatter(ells,real_conved, label='Realization', color='red', s=10)
        ax1.scatter(ells, pred_conved, label='Prediction', color='blue', s=10)
        ax1.set_ylabel(r'$D_{\ell}^{TT} [\mu K^2]$')
        # ax1.set_ylabel(r'$\ell(\ell+1)C_\ell/(2\pi)$ $\;$ [$\mu K^2$]')
        ax1.set_ylim(-300, 6500)
        ax1.legend()

        if baseline == "theory":
            ax2.axhline(0, color='black', linestyle='--', linewidth=0.5, label='Theory')
            ax2.scatter(ells, (real_conved - ps_theory[:n_ells]), label='Realization', color='red', s=7.5)
            ax2.scatter(ells, (pred_conved - ps_theory[:n_ells]), label='Prediction', color='blue', s=7.5)
            # ax2.fill_between(ells, error_band[0][0][:n_ells], error_band[0][1][:n_ells], color='darkorange', alpha=0.5, label=r'1$\sigma$ Error')
            # ax2.fill_between(ells, error_band[1][0][:n_ells], error_band[1][1][:n_ells], color='orange', alpha=0.5, label=r'2$\sigma$ Error')
            ax2.set_ylabel(r'Theory $\Delta$')
            ax2.set_xlabel(r'$\ell$')
            # ax2.set_ylim(-1250,1250)
            ax2.legend(loc='upper right')
        elif baseline == "real":
            ax2.axhline(0, color='red', linestyle='--', linewidth=0.5, label='Realization')
            ax2.scatter(ells, (ps_theory[:n_ells] - real_conved), label='Theory', color='black', s=7.5)
            ax2.scatter(ells, (pred_conved - real_conved), label='Prediction', color='blue', s=7.5)
            # ax2.fill_between(ells, error_band[0][0][:n_ells], error_band[0][1][:n_ells], color='darkorange', alpha=0.5, label=r'1$\sigma$ Error')
            # ax2.fill_between(ells, error_band[1][0][:n_ells], error_band[1][1][:n_ells], color='orange', alpha=0.5, label=r'2$\sigma$ Error')
            ax2.set_ylabel(r'Realization $\Delta$')
            ax2.set_xlabel(r'$\ell$')
            ax2.set_ylim(-1250,1250)
            ax2.legend(loc='upper right')
        else:
            raise ValueError("Baseline must be 'real' or 'theory'")
        