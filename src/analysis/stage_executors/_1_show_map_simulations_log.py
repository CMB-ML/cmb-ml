from typing import List, Dict, Union
from typing import List, Dict, Union
import logging

import numpy as np

from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from omegaconf import DictConfig
import healpy as hp

from src.core import (
    BaseStageExecutor, 
    Split,
    Asset
    )
# from tqdm import tqdm
from src.core.asset_handlers.asset_handlers_base import Mover
from src.core.asset_handlers.healpy_map_handler import HealpyMap
from src.utils.planck_instrument import make_instrument, Instrument
from src.utils.planck_cmap import colombi1_cmap


logger = logging.getLogger(__name__)


class ShowSimsLogExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following stage_str must match the pipeline yaml
        super().__init__(cfg, stage_str="show_sims")

        self.out_cmb_figure: Asset = self.assets_out["cmb_map_render"]
        self.out_cmb_figure2: Asset = self.assets_out["cmb_map_render2"]
        self.out_obs_figure: Asset = self.assets_out["obs_map_render"]
        out_cmb_figure_handler: Mover
        out_obs_figure_handler: Mover

        self.in_cmb_map: Asset = self.assets_in["cmb_map"]
        self.in_obs_map: Asset = self.assets_in["obs_maps"]
        in_cmb_map_handler: HealpyMap
        in_obs_map_handler: HealpyMap

        self.instrument: Instrument = make_instrument(cfg=cfg)

        # Only produce visualizations for a subset of sims
        self.sim_ns = self.get_override_sim_ns(cfg.pipeline[self.stage_str].override_n_sims)
        self.min_max = self.get_plot_min_max(cfg.pipeline[self.stage_str].plot_min_max)
        
        # For Gnomonic view
        self.plot_rot = cfg.pipeline[self.stage_str].plot_rot
        self.gnom_plot_res = cfg.pipeline[self.stage_str].plot_gnom_res

    def get_override_sim_ns(self, sim_nums: Union[None, list, int]):
        # Returns either a list of sims, or None
        try:
            return list(range(sim_nums))
        except TypeError:
            return sim_nums

    def get_plot_min_max(self, min_max):
        if min_max is None:
            plot_min = plot_max = None
        elif isinstance(min_max, int):
            plot_min = -min_max
            plot_max = min_max
        elif isinstance(min_max, list):
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
        if self.sim_ns is None:
            sim_iter = split.iter_sims()
        else:
            sim_iter = self.sim_ns

        for sim in sim_iter:
            with self.name_tracker.set_context("sim_num", sim):
                self.process_sim()

    def process_sim(self) -> None:
        cmb_map = self.in_cmb_map.read() / 1e6

        # self.make_maps_per_field(cmb_map, det="cmb", out_asset=self.out_cmb_figure)
        self.make_maps_per_field(cmb_map, det="cmb", out_asset=self.out_cmb_figure2, log=False)
        # for freq in self.instrument.dets:
        #     with self.name_tracker.set_context("freq", freq):
        #         obs_map = self.in_obs_map.read()
        #         self.make_maps_per_field(obs_map, det=freq, out_asset=self.out_obs_figure)

    def make_maps_per_field(self, some_map, det, out_asset, log=True):
        # split = self.name_tracker.context['split']
        # sim_n = f"{self.name_tracker.context['sim_num']:0{self.cfg.file_system.sim_str_num_digits}d}"
        if det == "cmb":
            # title_start = "CMB Realization (Target)"
            title = "CMB"
            fields = self.cfg.scenario.map_fields
        else:
            # title_start = f"Observation, {det} GHz (Feature)"
            title = f"{det} GHz"
            fields = self.instrument.dets[det].fields
        for field_str in fields:
            with self.name_tracker.set_context("field", field_str):
                field_idx = {'I': 0, 'Q': 1, 'U': 2}[field_str]
                if log:
                    self.make_mollview_log(some_map[field_idx], title)
                else:
                    self.make_mollview(some_map[field_idx], title)
                with self.name_tracker.set_context("view", "moll"):
                    self.save_figure(out_asset)

    def save_figure(self, out_asset):
        fn = out_asset.path.name
        plt.savefig(fn)
        plt.close()
        out_asset.write(source_location=fn)

    def plot_params(self, title):
        return dict(
            xsize=2400,
            min=0,
            max=1,
            rot=self.plot_rot,
            cmap=colombi1_cmap,
            hold=True,
            title=title)

    def make_mollview(self, some_map, title):
        some_map = some_map * 1e4
        vmin = some_map.min()
        vmax = some_map.max()
        if np.abs(vmin) > np.abs(vmax):
            vmax = -vmin * 0.9
        else:
            vmin = -vmax * 0.9

        # fig, (ax1, ax2) = plt.figure(figsize=(8, 6))
        plot_params = self.plot_params(title)

        norm = cm.colors.Normalize(vmin=vmin, vmax=vmax)
        some_map_norm = norm(some_map)
        
        gs = gridspec.GridSpec(2, 1, height_ratios=(9, 0.2))
        fig = plt.figure(figsize=(10,5), dpi=200)

        ax1 = fig.add_axes([0.1, 0.2, 0.65, 0.8])
        hp.mollview(some_map_norm, **plot_params, cbar=False, sub=ax1)

        ax2 = fig.add_axes([0.05, 0.125, 0.9, 0.025])
        mappable = cm.ScalarMappable(norm=norm, cmap=colombi1_cmap)
        cb = plt.colorbar(mappable, cax=ax2, orientation='horizontal')
        cb.formatter.set_powerlimits((0, 0))
        cb.update_ticks()
        cb.set_label('$\\delta \\text{T} [\\times 10^{-4} \\mu \\text{K}_\\text{CMB}]$')
        plt.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05)

    # def make_mollview(self, some_map, title):
    #     some_map = some_map * 1e4
    #     vmin = some_map.min()
    #     vmax = some_map.max()
    #     if np.abs(vmin) > np.abs(vmax):
    #         vmax = -vmin * 0.9
    #     else:
    #         vmin = -vmax * 0.9

    #     # fig, (ax1, ax2) = plt.figure(figsize=(8, 6))
    #     plot_params = self.plot_params(title)

    #     norm = cm.colors.Normalize(vmin=vmin, vmax=vmax)
    #     some_map_norm = norm(some_map)
        
    #     gs = gridspec.GridSpec(1, 2, width_ratios=(9, 0.2))
    #     fig = plt.figure(figsize=(10,5), dpi=200)

    #     ax1 = fig.add_axes([0.1, 0.1, 0.65, 0.8])
    #     hp.mollview(some_map_norm, **plot_params, cbar=False, sub=ax1)

    #     ax2 = fig.add_axes([0.80, 0.2, 0.0125, 0.6])
    #     mappable = cm.ScalarMappable(norm=norm, cmap=planck_cmap.colombi1_cmap)
    #     cb = plt.colorbar(mappable, cax=ax2, orientation='vertical')
    #     cb.formatter.set_powerlimits((0, 0))
    #     cb.update_ticks()
    #     cb.set_label('$\\delta \\text{T} [\\times 10^{-4} \\mu \\text{K}_\\text{CMB}]$')
    #     plt.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05)

    def make_mollview_log(self, some_map, title):
        """
        For observation maps
        """

        linthresh = 1e-4
        linscale = 0.5
        vmin = -0.006
        vmax = 1.5e3

        # fig, (ax1, ax2) = plt.figure(figsize=(8, 6))
        plot_params = self.plot_params(title)

        norm = cm.colors.SymLogNorm(vmin=vmin, vmax=vmax, linthresh=linthresh, linscale=linscale)
        some_map_norm = norm(some_map)

        gs = gridspec.GridSpec(2, 1, height_ratios=(9, 0.2))
        # gs = gridspec.GridSpec(1, 2, width_ratios=(9, 0.2))
        fig = plt.figure(figsize=(10,5), dpi=200)

        ax1 = fig.add_axes([0.1, 0.2, 0.65, 0.8])
        hp.mollview(some_map_norm, **plot_params, cbar=False, sub=ax1)

        ax2 = fig.add_axes([0.05, 0.125, 0.9, 0.025])
        mappable = cm.ScalarMappable(norm=norm, cmap=colombi1_cmap)
        cb = plt.colorbar(mappable, cax=ax2, orientation='horizontal')
        cb.update_ticks()
        cb.set_label('$\\delta \\text{T} \\; [\\mu \\text{K}_\\text{CMB}]$')
        plt.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05)

    # def make_mollview_log(self, some_map, title):
    #     """
    #     For observation maps
    #     """

    #     linthresh = 1e-4
    #     linscale = 0.5
    #     vmin = -0.006
    #     vmax = 1.5e3

    #     # fig, (ax1, ax2) = plt.figure(figsize=(8, 6))
    #     plot_params = self.plot_params(title)

    #     norm = cm.colors.SymLogNorm(vmin=vmin, vmax=vmax, linthresh=linthresh, linscale=linscale)
    #     some_map_norm = norm(some_map)

    #     # gs = gridspec.GridSpec(2, 1, height_ratios=(9, 0.2))
    #     gs = gridspec.GridSpec(1, 2, width_ratios=(9, 0.2))
    #     fig = plt.figure(figsize=(10,5), dpi=200)

    #     ax1 = fig.add_axes([0.1, 0.1, 0.65, 0.8])
    #     hp.mollview(some_map_norm, **plot_params, cbar=False, sub=ax1)

    #     ax2 = fig.add_axes([0.80, 0.2, 0.0125, 0.6])
    #     mappable = cm.ScalarMappable(norm=norm, cmap=planck_cmap.colombi1_cmap)
    #     cb = plt.colorbar(mappable, cax=ax2, orientation='vertical')
    #     cb.update_ticks()
    #     cb.set_label('$\\delta \\text{T} \\; [\\mu \\text{K}_\\text{CMB}]$')
    #     plt.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05)
