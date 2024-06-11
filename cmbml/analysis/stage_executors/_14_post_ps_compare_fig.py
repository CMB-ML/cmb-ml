from typing import Union
import logging

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from omegaconf import DictConfig, OmegaConf

from scipy import stats

import json
from tqdm import tqdm

from cmbml.core import (
    BaseStageExecutor, 
    Split,
    Asset, AssetWithPathAlts
    )

from cmbml.core.asset_handlers.asset_handlers_base import EmptyHandler # Import for typing hint
from cmbml.core.asset_handlers.psmaker_handler import NumpyPowerSpectrum


logger = logging.getLogger(__name__)


BLACK = 'black'
RED = "#ED1C24"
PURBLUE = "#524FA1"
GREEN = "#00A651"
YELLOW = "#FDB913"


class PostAnalysisPsCompareFigExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="post_post_ps_fig")

        self.out_ps_figure_theory: Asset = self.assets_out["ps_figure_theory"]
        # self.out_ps_figure_real: Asset = self.assets_out["ps_figure_real"]
        out_ps_figure_handler: EmptyHandler

        self.in_ps_theory: AssetWithPathAlts = self.assets_in["theory_ps"]
        self.in_ps_real: Asset = self.assets_in["auto_real"]
        self.in_ps_pred: Asset = self.assets_in["auto_pred"]
        self.in_wmap_ave: Asset = self.assets_in["wmap_ave"]
        self.in_wmap_std: Asset = self.assets_in["wmap_std"]
        # self.in_wmap_distribution: Asset = self.assets_in["wmap_distribution"]
        # self.in_error_distribution: Asset = self.assets_in["error_distribution"]
        in_ps_handler: NumpyPowerSpectrum

        self.nside = self.cfg.scenario.nside
        self.lmax = int(cfg.model.analysis.lmax_ratio * self.nside) - 2

        self.fig_model_name = cfg.fig_model_name
        self.models_to_compare = cfg.models_comp

    def execute(self) -> None:
        # Remove this function
        logger.debug(f"Running {self.__class__.__name__} execute()")
        for split in self.splits:
            with self.name_tracker.set_context("split", split.name):
                self.process_split(split)

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
        spectra = []
        for model_comp in self.models_to_compare:
            epochs = model_comp.get("epochs", None)
            if epochs is None:
                epoch = model_comp.get("epoch")
            else:
                # Hard code to just do the last epoch. TODO: Fix this!
                epoch = epochs[-1]
            with self.name_tracker.set_context('epoch', epoch):
                ps = self.get_ps_for(model_comp)
            res = dict(
                model=model_comp,
                epochs=epochs,
                ps=ps
            )
            spectra.append(res)
        self.make_figure(ps_theory, spectra)

    def get_ps_for(self, model_comp):
        model_dict = OmegaConf.to_container(model_comp, resolve=True)
        working_directory = model_dict["working_directory"]

        with self.name_tracker.set_context("working", working_directory):
            ps = self.in_ps_pred.read()
        return ps

    def make_figure(self, ps_theory, spectra) -> None:
        if ps_theory is None:
            ps_theory = self.in_ps_theory.read(use_alt_path=False)

        wmap_ave = self.in_wmap_ave.read()
        wmap_std = self.in_wmap_std.read()

        wmap_band = [(wmap_ave - wmap_std, wmap_ave + wmap_std), (wmap_ave - 2*wmap_std, wmap_ave + 2*wmap_std)]

        fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 6))
        ax1, ax2 = axs

        colors = [PURBLUE, RED, YELLOW, GREEN ]
        label_xlate = {'CMBNNCS': 'CMBNNCS Prediction', 'CNILC': 'CNILC Prediction'}

        # fig, ax = plt.subplots()
        # fig.suptitle('Marker fillstyle', fontsize=14)
        # fig.subplots_adjust(left=0.4)

        # filled_marker_style = dict(marker='o', linestyle=':', markersize=15,
        #                         color='darkgrey',
        #                         markerfacecolor='tab:blue',
        #                         markerfacecoloralt='lightsteelblue',
        #                         markeredgecolor='brown')

        # for y, fill_style in enumerate(Line2D.fillStyles):
        #     ax.text(-0.5, y, repr(fill_style), **text_style)
        #     ax.plot([y] * 3, fillstyle=fill_style, **filled_marker_style)
        # format_axes(ax)

        n_ells = self.lmax
        ells = np.arange(1, n_ells+1)
        ells = ells[2:n_ells]
        thry_params = dict(color=BLACK, label='Theory')
        wmap1_params = dict(color=GREEN, alpha=0.25, label='1$\\sigma$ WMAP')
        wmap2_params = dict(color=GREEN, alpha=0.50, label='2$\\sigma$ WMAP')
        horz_params = dict(linestyle='--', linewidth=0.5, label='Theory', color=BLACK)
        # horz_params = {'label': 'Theory', 'color': BLACK, **horz_params}
        wmap1_lower = wmap_band[0][0][2:n_ells]
        wmap1_upper = wmap_band[0][1][2:n_ells]
        wmap2_lower = wmap_band[1][0][2:n_ells]
        wmap2_upper = wmap_band[1][1][2:n_ells]
        ps_theory = ps_theory[2:n_ells]

        ax1.fill_between(ells, wmap1_lower, wmap1_upper, **wmap1_params)
        ax1.fill_between(ells, wmap2_lower, wmap2_upper, **wmap2_params)
        ax1.plot(ells, ps_theory, **thry_params)

        ax2.axhline(0, **horz_params)

        # TODO: temporary. Better to specify colors per model in the configs and use those; 
        # can have fixed presentation style
        fillstyles = ['left', 'right']
        for spec, color, fillstyle in zip(spectra, colors, fillstyles):
            model_name = spec['model']['model_name']
            pred_params = dict(color=color, label=label_xlate[model_name])
            ps_pred = spec['ps'][2:n_ells]
            self.add_ps_to_figures(axs, ells, ps_pred, ps_theory, pred_params, fillstyle)

        ax1.set_ylabel('$D_{\ell}^\\text{TT} [\\mu \\text{K}^2]$')
        ax1.set_ylim(-300, 6500)
        ax1.legend(markerscale=3)

        ax2.set_xlabel('$\\ell$')

        ax2.set_ylabel('$\\% \\Delta D_{\ell}^\\text{TT} [\\mu \\text{K}^2]$')
        ax2.set_ylim(-50, 50)

        # title = self.make_title(epoch, split, sim_num)
        # plt.suptitle(title)
        self.out_ps_figure_theory.write()
        fn = self.out_ps_figure_theory.path
        print(f'writing to {fn}')
        plt.savefig(fn)
        plt.close()

    def add_ps_to_figures(self, axs, ells, ps_pred, ps_theory, pred_params, fillstyle):
        ax1, ax2 = axs

        # Upper Panel
        self.abs_panel(ax1, ells, ps_pred, pred_params)

        # Lower Panel
        deltas = (ps_pred - ps_theory) / ps_theory * 100
        self.rel_panel(ax2, ells, deltas, pred_params, fillstyle)

    def abs_panel(self, ax, ells, pred_conved, pred_params):
        # Upper panel
        marker_size = 1
        ax.scatter(ells, pred_conved, s=marker_size, **pred_params)

    def rel_panel(self, ax, ells, deltas, deltas_params, fillstyle):
        # Lower panel
        marker_size = 5
        bin_width = 30

        bin_centers2, binned_means2, binned_stds2 = self.bin_data(ells, deltas, bin_width)
        ax.errorbar(bin_centers2, binned_means2, yerr=binned_stds2, fmt='o', markersize=marker_size, fillstyle=fillstyle, markeredgecolor='none', **deltas_params)

    def bin_data(self, ells, deltas, bin_width):
        # Calculate the bin edges
        bin_edges = np.arange(min(ells), max(ells) + bin_width, bin_width)
        # Digitize the ells data to find out which bin each value belongs to
        bin_indices = np.digitize(ells, bin_edges)
        # Calculate the mean and standard deviation of deltas values within each bin
        binned_means = []
        binned_stds = []
        for i in range(1, len(bin_edges)):
            bin_values = deltas[bin_indices == i]
            binned_means.append(np.mean(bin_values))
            binned_stds.append(np.std(bin_values))
        # Calculate the center of each bin for plotting purposes
        bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
        return bin_centers, binned_means, binned_stds

    def make_title(self, epoch, split, sim_num):
        if epoch != "":
            e_phrase = f" After Training for {epoch} Epochs"
        else:
            e_phrase = ""
        return f"{self.fig_model_name} Predictions{e_phrase}, {split}:{sim_num}"
