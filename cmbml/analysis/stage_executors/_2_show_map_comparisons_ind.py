"""
Produce, for each specified simulation, the target CMB, the prediction, the difference, and a histogram of the difference.

Each figure will be in its own file.
"""
from typing import List, Dict, Union
import logging

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from omegaconf import DictConfig, ListConfig
import healpy as hp
import pysm3.units as u

from cmbml.core import (
    BaseStageExecutor, 
    Split,
    Asset
    )
from cmbml.core.asset_handlers import Config
from cmbml.core.asset_handlers.asset_handlers_base import EmptyHandler
from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap
from cmbml.core.asset_handlers.qtable_handler import QTableHandler  # Import to register handler

from cmbml.core.asset_handlers.handler_npymap import NumpyMap
from cmbml.utils.planck_instrument import make_instrument, Instrument
from cmbml.utils import planck_cmap


logger = logging.getLogger(__name__)


class ShowSimsPostIndivExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig, stage_str=None) -> None:
        stage_str = "show_cmb_post_masked_ind"
        super().__init__(cfg, stage_str)

        in_det_table: Asset  = self.assets_in['deltabandpass']
        in_det_table_handler: QTableHandler
        det_info = in_det_table.read()
        self.instrument: Instrument = make_instrument(cfg=cfg, det_info=det_info)

        if self.override_sim_nums is None:
            logger.warning("No particular sim indices specified. Outputs will be produced for all. This is not recommended.")
        self.min_max = self.get_plot_min_max()
        self.fig_model_name = cfg.fig_model_name

        self.suptitle = "CMB Predictions"
        self.right_subplot_title = "Predicted"

        self.out_real_fig: Asset = self.assets_out["real_map_render"]
        self.out_pred_fig: Asset = self.assets_out["pred_map_render"]
        self.out_diff_fig: Asset = self.assets_out["diff_map_render"]
        self.out_hist_fig: Asset = self.assets_out["hist_render"]
        self.out_real_pred_cbar: Asset = self.assets_out["real_pred_cbar"]
        self.out_diff_cbar: Asset = self.assets_out["diff_cbar"]
        out_figure_handler: EmptyHandler

        self.in_cmb_map_post: Asset = self.assets_in["cmb_map_post"]
        self.in_cmb_map_sim: Asset = self.assets_in["cmb_map_sim"]
        in_cmb_map_handler: HealpyMap

        self.diff_min_max = [-120, 120]
        self.hist_color = "#524FA1"

        self.plot_size_map = (3, 1.5)
        self.plot_size_hist = (3, 1.5)
        self.cbar_height = 0.1
        self.cbar_shrink = 0.7

    def get_plot_min_max(self):
        """
        Handles reading the minimum intensity and maximum intensity from cfg files
        TODO: Better docstring
        """
        min_max = self.get_stage_element("plot_min_max")
        if min_max is None:
            plot_min = plot_max = None
        elif isinstance(min_max, int):
            plot_min = -min_max
            plot_max = min_max
        elif isinstance(min_max, ListConfig):
            plot_min = min_max[0]
            plot_max = min_max[1]
        return plot_min, plot_max

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

    def process_sim(self) -> None:
        for epoch in self.model_epochs:
            with self.name_tracker.set_context('epoch', epoch):
                cmb_map_sim = self.in_cmb_map_sim.read()
                cmb_map_post = self.in_cmb_map_post.read()
                self.make_maps_per_field(cmb_map_sim, 
                                         cmb_map_post)

    def make_colorbars(self):
        figure_width = self.plot_size_map[0] * 2       # Width of figure for Real + Pred map colorbar
        figure_width = figure_width * self.cbar_shrink
        init_height = 2                                # Ample height for figure, to crop in LaTeX
        cbar_aspect = figure_width / self.cbar_height  # Aspect ratio of the colorbar

        fig, ax = plt.subplots(figsize=(figure_width, init_height))
        norm = plt.Normalize(vmin=self.min_max[0], vmax=self.min_max[1])
        sm = plt.cm.ScalarMappable(cmap=planck_cmap.colombi1_cmap, norm=norm)
        sm.set_array([])                               # Dummy array for ScalarMappable
        cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", aspect=cbar_aspect)
        cbar.set_ticks(self.min_max)
        cbar.set_label("$\\delta \\mathrm{T} \\; [\\mu \\mathrm{K}_\\mathrm{CMB}]$", labelpad=-10)
        fig.delaxes(ax)                                # Remove the Axes needed for making the colorbar
        self.save_figure(self.out_real_pred_cbar)

        figure_width = self.plot_size_map[0]           # Width of figure for Diff map colorbar
        figure_width = figure_width * self.cbar_shrink
        init_height = 2                                # Ample height for figure, to crop in LaTeX
        cbar_aspect = figure_width / self.cbar_height  # Aspect ratio of the colorbar

        fig, ax = plt.subplots(figsize=(figure_width, init_height))
        norm = plt.Normalize(vmin=self.diff_min_max[0], vmax=self.diff_min_max[1])
        sm = plt.cm.ScalarMappable(cmap=planck_cmap.colombi1_cmap, norm=norm)
        sm.set_array([])                               # Dummy array for ScalarMappable
        cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", aspect=cbar_aspect)
        cbar.set_ticks(self.diff_min_max)
        cbar.set_label("$\\delta \\mathrm{T} \\; [\\mu \\mathrm{K}_\\mathrm{CMB}]$", labelpad=-10)
        fig.delaxes(ax)                                # Remove the Axes needed for making the colorbar
        self.save_figure(self.out_diff_cbar)

    def make_maps_per_field(self, map_sim, map_post):
        """
        Makes a figure for each field in the maps (e.g., IQU will result in 3 figures)
        """
        split = self.name_tracker.context['split']
        sim_n = f"{self.name_tracker.context['sim_num']:0{self.cfg.file_system.sim_str_num_digits}d}"

        fields = self.cfg.scenario.map_fields

        for field_str in fields:
            with self.name_tracker.set_context("field", field_str):
                if sim_n == f"{0:0{self.cfg.file_system.sim_str_num_digits}d}":
                    self.make_colorbars()
                field_idx = {'I': 0, 'Q': 1, 'U': 2}[field_str]
                mask = map_sim[field_idx] == hp.UNSEEN
                diff = map_post[field_idx] - map_sim[field_idx]
                diff = hp.ma(diff)
                diff.mask = mask

                diff_min, diff_max = self.diff_min_max
                self.make_mollview(map_sim[field_idx], title="Realization", out_asset=self.out_real_fig)
                self.make_mollview(map_post[field_idx], title="Prediction", out_asset=self.out_pred_fig)
                self.make_mollview(diff, title="Difference", out_asset=self.out_diff_fig, min_or=diff_min, max_or=diff_max)

                # healpy applies the graticule to every subplot, so we only need to do it once; 
                # if this were in self.make_mollview, earlier subplots will have multiple graticules applied.
                hp.graticule(dpar=45, dmer=45)

                n_bins = 50

                fig, ax = plt.subplots(1, 1, figsize=self.plot_size_hist)  # DPI doesn't matter; exporting vector graphic
                plt.hist(diff.compressed(), bins=n_bins, range=(diff_min, diff_max), color=self.hist_color, histtype='stepfilled')
                ax.set_yticks([])
                ax.set_xticks([])
                # Enable once to get ticks for figure in Supplementary Material. TODO: Better way.
                # ax.set_xticks([-100, -50, 0, 50, 100])
                # ax.set_xlabel("$\\delta \\mathrm{T} \\; [\\mu \\mathrm{K}_\\mathrm{CMB}]$")
                # ax.set_ylabel("Pixel Count")
                # ax.set_title("Histogram of Difference")
                for x in [-100, -50, 0, 50, 100]:  # Hard-coded. Sorry. :(
                    ax.axvline(x=x, color='black', linestyle='--', linewidth=0.5)
                plt.tight_layout()
                self.save_figure(self.out_hist_fig)

    def save_figure(self, out_asset):
        fn = out_asset.path
        out_asset.write()  # Create directory if it doesn't exist
        file_fmt = fn.suffix[1:]  # Remove the dot
        plt.savefig(fn, format=file_fmt)
        plt.close()

    def make_mollview(self, some_map, title, out_asset, min_or=None, max_or=None):
        fig = plt.figure(figsize=self.plot_size_map, dpi=300)
        to_plot = some_map
        vmin = self.min_max[0] if min_or is None else min_or
        vmax = self.min_max[1] if max_or is None else max_or
        plot_params = dict(
            xsize=2400,
            min=vmin, 
            max=vmax,
            title="",
            cmap=planck_cmap.colombi1_cmap,
            hold=True,
            cbar=False,
            margins=(-0.125, -0.1, -0.1, -0.1),  # Left, right, bottom, top; healpy is difficult.
        )
        hp.mollview(to_plot, **plot_params)
        self.save_figure(out_asset)


class CommonRealPostIndivExecutor(ShowSimsPostIndivExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        stage_str = "show_cmb_post_masked_ind"
        super().__init__(cfg, stage_str=stage_str)


class CommonCMBNNCSShowSimsPostIndivExecutor(CommonRealPostIndivExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.right_subplot_title = "CMBNNCS Predicted"


class CommonPetroffShowSimsPostIndivExecutor(CommonRealPostIndivExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.right_subplot_title = "Petroff Predicted"


class CommonNILCShowSimsPostIndivExecutor(CommonRealPostIndivExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.right_subplot_title = "NILC Predicted"
        self.suptitle = cfg.fig_model_name
