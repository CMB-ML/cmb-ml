from typing import List, Dict, Union
import logging

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from omegaconf import DictConfig, ListConfig
import healpy as hp

from cmbml.core import (
    BaseStageExecutor, 
    Split,
    Asset, AssetWithPathAlts
    )
from cmbml.core.asset_handlers.asset_handlers_base import EmptyHandler
from cmbml.core.asset_handlers.ps_handler import NumpyPowerSpectrum


logger = logging.getLogger(__name__)


BLACK = 'black'
RED = "#ED1C24"
PURBLUE = "#524FA1"
GREEN = "#00A651"
YELLOW = "#FDB913"


class ShowOnePSExecutor(BaseStageExecutor):
    """
    Makes a single Power Spectrum plot for each split in the pipeline yaml.
    
    Will only make a subset of test set as images given by `override_n_sims`
        in the pipeline yaml for this stage.
    """
    def __init__(self, cfg: DictConfig) -> None:
        # The following stage_str must match the pipeline yaml
        super().__init__(cfg, stage_str="one_ps_fig")

        self.out_ps_figure: Asset = self.assets_out["ps_figure"]
        out_ps_figure_handler: EmptyHandler

        self.in_ps_theory: AssetWithPathAlts = self.assets_in["theory_ps"]
        self.in_ps_real: Asset = self.assets_in["auto_real"]
        self.in_ps_pred: Asset = self.assets_in["auto_pred"]
        in_ps_handler: NumpyPowerSpectrum

        self.fig_model_name = cfg.fig_model_name
        if self.override_sim_nums is None:
            logger.warning("No particular sim indices specified. Outputs will be produced for all. This is not recommended.")

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute().")
        self.default_execute()

    def process_split(self, 
                      split: Split) -> None:
        logger.info(f"Running {self.__class__.__name__} process_split() for split: {split.name}.")

        # We may want to process a subset of all sims
        if self.override_sim_nums is None:
            sim_iter = split.iter_sims()
        else:
            sim_iter = self.override_sim_nums

        for sim in sim_iter:
        # for sim in tqdm(sim_iter):
            with self.name_tracker.set_context("sim_num", sim):
                self.process_sim()

    def process_sim(self):
        for epoch in self.model_epochs:
            with self.name_tracker.set_context("epoch", epoch):
                self.process_epoch()

    def process_epoch(self) -> None:
        epoch = self.name_tracker.context['epoch']
        split = self.name_tracker.context['split']
        sim_num = self.name_tracker.context['sim_num']
        ps_real = self.in_ps_real.read()
        ps_pred = self.in_ps_pred.read()

        # if ps_theory is None:
        ps_theory = self.in_ps_theory.read(use_alt_path=False)

        self.make_ps_figure(ps_real, ps_pred, ps_theory, baseline="theory")
        title = self.make_title(epoch, split, sim_num)
        plt.suptitle(title)
        self.out_ps_figure.write()
        fn = self.out_ps_figure.path
        print(f'writing to {fn}')
        plt.savefig(fn)
        plt.close()

    def make_ps_figure(self, ps_real, ps_pred, ps_theory, baseline="theory"):
        n_ells = ps_real.shape[0] - 2
        ells = np.arange(1, n_ells+1)

        pred_conved = ps_pred
        real_conved = ps_real

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 6))

        thry_params = dict(color=BLACK, label='Theory')
        # real_params = dict(color=RED, label='Realization')
        pred_params = dict(color=PURBLUE, label='Prediction')

        ells = ells[2:n_ells]
        ps_theory = ps_theory[2:n_ells]
        # real_conved = real_conved[2:n_ells]
        pred_conved = pred_conved[2:n_ells]
        self.abs_panel(ax1, ells, pred_conved, ps_theory, thry_params, pred_params)
        horz_params = dict(linestyle='--', linewidth=0.5)
        if baseline == "theory":
            horz_params = {'label': 'Theory', 'color': BLACK, **horz_params}
            # deltas1 = (real_conved - ps_theory) / ps_theory * 100
            # deltas1_params = real_params
            deltas = (pred_conved - ps_theory) / ps_theory * 100
            deltas_params = pred_params
        elif baseline == "real":
            horz_params = {'label': 'Realization', 'color': RED, **horz_params}
            # deltas1 = ps_theory - real_conved
            # deltas1_params = thry_params
            deltas = pred_conved - real_conved
            deltas_params = pred_params
        else:
            raise ValueError("Baseline must be 'real' or 'theory'")
        ylabel = '$\\%\\Delta D_{\ell}^\\text{TT} [\\mu \\text{K}^2]$'
        self.rel_panel(ax2, ells, ylabel, deltas, 
                       deltas_params, horz_params)

    def abs_panel(self, ax, ells, pred_conved, ps_theory, thry_params, pred_params):
        # TODO: refactor
        # Upper panel
        marker_size = 5
        ax.plot(ells, ps_theory, **thry_params)
        # ax.scatter(ells,real_conved, s=marker_size, **real_params)
        ax.scatter(ells, pred_conved, s=marker_size, **pred_params)
        ax.set_ylabel('$D_{\ell}^\\text{TT} [\\mu \\text{K}^2]$')
        # ax1.set_ylabel(r'$\ell(\ell+1)C_\ell/(2\pi)$ $\;$ [$\mu K^2$]')
        ax.set_ylim(-300, 6500)
        ax.legend()

    def rel_panel(self, ax, ells, ylabel, deltas, 
                  deltas_params, horz_params):
        # TODO: refactor
        # Lower panel
        marker_size = 5
        bin_width = 30

        # bin_centers1, binned_means1, binned_stds1 = self.bin_data(ells, deltas1, bin_width)
        bin_centers2, binned_means2, binned_stds2 = self.bin_data(ells, deltas, bin_width)

        # Lower panel
        ax.axhline(0, **horz_params)
        # ax.errorbar(bin_centers1, binned_means1, yerr=binned_stds1, fmt='o', markersize=marker_size, **deltas1_params)
        ax.errorbar(bin_centers2, binned_means2, yerr=binned_stds2, fmt='o', markersize=marker_size, **deltas_params)
        ax.set_xlabel('$\\ell$')
        ax.set_ylabel(ylabel)
        ax.set_ylim(-50, 50)
        # ax.legend(loc='upper right')

    def bin_data(self, ells, deltas, bin_width):
        # Calculate the bin edges
        bin_edges = np.arange(min(ells), max(ells) + bin_width, bin_width)
        
        # Digitize the ells data to find out which bin each value belongs to
        bin_indices = np.digitize(ells, bin_edges)
        
        # Calculate the mean and standard deviation of deltas values within each bin, applying weights
        binned_means = []
        binned_stds = []
        
        for i in range(1, len(bin_edges)):
            bin_values = deltas[bin_indices == i]
            bin_ells = ells[bin_indices == i]
            
            # Apply weights: (2 * ell + 1)
            weights = 2 * bin_ells + 1
            if len(bin_values) > 0:
                weighted_mean = np.average(bin_values, weights=weights)
                weighted_var = np.average((bin_values - weighted_mean)**2, weights=weights)
                
                binned_means.append(weighted_mean)
                binned_stds.append(np.sqrt(weighted_var))
            else:
                binned_means.append(np.nan)
                binned_stds.append(np.nan)
        
        # Calculate the center of each bin for plotting purposes
        bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
        return bin_centers, binned_means, binned_stds


    def make_title(self, epoch, split, sim_num):
        if epoch != "":
            e_phrase = f" After Training for {epoch} Epochs"
        else:
            e_phrase = ""
        return f"{self.fig_model_name} Predictions{e_phrase}, {split}:{sim_num}"


    def save_figure(self, title, split_name, sim_num, field_str, out_asset):
        plt.suptitle(f"{title}, {split_name}:{sim_num} {field_str} Stokes")

        fn = out_asset.path
        out_asset.write()
        plt.savefig(fn)
        plt.close()
