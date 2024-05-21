from typing import List, Dict, Union
import logging

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from omegaconf import DictConfig, ListConfig
import healpy as hp

from core import (
    BaseStageExecutor, 
    Split,
    Asset
    )
from core.asset_handlers.asset_handlers_base import Mover
from core.asset_handlers.healpy_map_handler import HealpyMap

from src.cmbnncs_local.handler_npymap import NumpyMap
from utils.planck_instrument import make_instrument, Instrument
from utils import planck_cmap


logger = logging.getLogger(__name__)


class ShowSimsExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig, stage_str: str) -> None:
        # The following stage_str must match the pipeline yaml
        super().__init__(cfg, stage_str=stage_str)

        if self.__class__.__name__ == "ShowSimsExecutor":
            # TODO: Can I ABC this?
            raise NotImplementedError("This is a base class. Kinda. Sorta. Not sure if I can ABC this.")

        self.instrument: Instrument = make_instrument(cfg=cfg)

        # Only produce visualizations for a subset of sims

        if self.override_sim_nums is None:
            logger.warning("No particular sim indices specified. Outputs will be produced for all. This is not recommended.")
        self.min_max = self.get_plot_min_max()

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

    def process_sim(self):
        raise NotImplementedError("This is intended to be an abstract class. process_sim() should be overwritten.")

    def make_maps_per_field(self, map_sim, map_prep, det, out_asset):
        split = self.name_tracker.context['split']
        sim_n = f"{self.name_tracker.context['sim_num']:0{self.cfg.file_system.sim_str_num_digits}d}"
        if det == "cmb":
            title_start = "CMB Realization (Target)"
            fields = self.cfg.scenario.map_fields
        else:
            title_start = f"Observation, {det} GHz"
            fields = self.instrument.dets[det].fields
        for field_str in fields:
            with self.name_tracker.set_context("field", field_str):
                field_idx = {'I': 0, 'Q': 1, 'U': 2}[field_str]
                fig = plt.figure(figsize=(12, 6))
                gs = gridspec.GridSpec(1, 3, width_ratios=[6, 3, 0.1], wspace=0.1)

                (ax1, ax2, cbar_ax) = [plt.subplot(gs[i]) for i in [0,1,2]]

                self.make_mollview(map_sim[field_idx], ax1)
                self.make_imshow(map_prep[field_idx], ax2)

                norm = plt.Normalize(vmin=self.min_max[0], vmax=self.min_max[1])
                sm = plt.cm.ScalarMappable(cmap=planck_cmap.colombi1_cmap, norm=norm)
                sm.set_array([])
                fig.colorbar(sm, cax=cbar_ax)

                self.save_figure(title_start, split, sim_n, field_str, out_asset)

    def save_figure(self, title, split_name, sim_num, field_str, out_asset):
        plt.suptitle(f"{title}, {split_name}:{sim_num} {field_str} Stokes")

        fn = out_asset.path.name
        plt.savefig(fn)
        plt.close()
        out_asset.write(source_location=fn)

    def make_imshow(self, some_map, ax):
        plt.axes(ax)
        plot_params = dict(
            vmin=self.min_max[0],
            vmax=self.min_max[1],
            cmap=planck_cmap.colombi1_cmap,
        )
        plt.imshow(some_map, **plot_params)
        plt.title(self.right_subplot_title)
        ax.set_axis_off()
        plt.colorbar

    def make_mollview(self, some_map, ax, show_cbar=False, title="Raw Simulation"):
        plt.axes(ax)
        plot_params = dict(
            min=self.min_max[0], 
            max=self.min_max[1],
            cmap=planck_cmap.colombi1_cmap,
            hold=True,
            cbar=show_cbar
        )
        hp.mollview(some_map, **plot_params)
        plt.title(title)


class ShowSimsPrepExecutor(ShowSimsExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        stage_str = "show_sims_prep_cmbnncs"
        super().__init__(cfg, stage_str)

        self.right_subplot_title = "Preprocessed"

        self.out_cmb_figure: Asset = self.assets_out["cmb_map_render"]
        self.out_obs_figure: Asset = self.assets_out["obs_map_render"]
        out_cmb_figure_handler: Mover
        out_obs_figure_handler: Mover

        self.in_cmb_map_sim: Asset = self.assets_in["cmb_map_sim"]
        self.in_cmb_map_prep: Asset = self.assets_in["cmb_map_prep"]
        self.in_obs_map_sim: Asset = self.assets_in["obs_maps_sim"]
        self.in_obs_map_prep: Asset = self.assets_in["obs_maps_prep"]
        in_cmb_map_handler: NumpyMap
        in_obs_map_handler: NumpyMap

    def process_sim(self) -> None:
        cmb_map_sim = self.in_cmb_map_sim.read()
        cmb_map_prep = self.in_cmb_map_prep.read()
        self.make_maps_per_field(cmb_map_sim, cmb_map_prep, det="cmb", out_asset=self.out_cmb_figure)
        for freq in self.instrument.dets:
            with self.name_tracker.set_context("freq", freq):
                obs_map_sim = self.in_obs_map_sim.read()
                obs_map_prep = self.in_obs_map_prep.read()
                self.make_maps_per_field(obs_map_sim, obs_map_prep, det=freq, out_asset=self.out_obs_figure)


class CMBNNCSShowSimsPredExecutor(ShowSimsExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        stage_str = "show_sims_pred_cmbnncs"
        super().__init__(cfg, stage_str)

        self.right_subplot_title = "Predicted"

        self.out_cmb_figure: Asset = self.assets_out["cmb_map_render"]
        out_cmb_figure_handler: Mover

        self.in_cmb_map_sim: Asset = self.assets_in["cmb_map_sim"]
        self.in_cmb_map_pred: Asset = self.assets_in["cmb_map_pred"]
        in_cmb_map_sim_handler: HealpyMap
        in_cmb_map_pred_handler: NumpyMap

    def process_sim(self) -> None:
        for epoch in self.model_epochs:
            logger.info(f"Creating map figures predictions, model epoch {epoch}")
            with self.name_tracker.set_context('epoch', epoch):
                cmb_map_sim = self.in_cmb_map_sim.read()
                cmb_map_prep = self.in_cmb_map_pred.read()
                self.make_maps_per_field(cmb_map_sim, 
                                         cmb_map_prep, 
                                         det="cmb",
                                         out_asset=self.out_cmb_figure)


class ShowSimsPostExecutor(ShowSimsExecutor):
    def __init__(self, cfg: DictConfig, stage_str=None) -> None:
        if stage_str is None:
            stage_str = "show_sims_post"
        super().__init__(cfg, stage_str)

        self.right_subplot_title = "Predicted"

        self.out_cmb_figure: Asset = self.assets_out["cmb_map_render"]
        out_cmb_figure_handler: Mover

        self.in_cmb_map_sim: Asset = self.assets_in["cmb_map_sim"]
        in_cmb_map_handler: HealpyMap

    def process_sim(self) -> None:
        for epoch in self.model_epochs:
            with self.name_tracker.set_context('epoch', epoch):
                cmb_map_sim = self.in_cmb_map_sim.read()
                cmb_map_post = self.in_cmb_map.read()
                self.make_maps_per_field(cmb_map_sim, 
                                         cmb_map_post, 
                                         out_asset=self.out_cmb_figure)

    def make_maps_per_field(self, map_sim, map_post, out_asset):
        split = self.name_tracker.context['split']
        sim_n = f"{self.name_tracker.context['sim_num']:0{self.cfg.file_system.sim_str_num_digits}d}"
        fields = self.cfg.scenario.map_fields

        for field_str in fields:
            with self.name_tracker.set_context("field", field_str):
                field_idx = {'I': 0, 'Q': 1, 'U': 2}[field_str]
                fig = plt.figure(figsize=(12, 6))
                gs = gridspec.GridSpec(1, 2, width_ratios=[6, 6], wspace=0.1)

                (ax1, ax2) = [plt.subplot(gs[i]) for i in [0,1]]

                self.make_mollview(map_sim[field_idx], ax1, show_cbar=True, title="Realization")
                self.make_mollview(map_post[field_idx], ax2, show_cbar=True, title="Prediction")

                self.save_figure("CMB Predictions", split, sim_n, field_str, out_asset)


class CMBNNCSShowSimsPostExecutor(ShowSimsPostExecutor):
    def __init__(self, cfg: DictConfig, stage_str=None) -> None:
        if stage_str is None:
            stage_str = "show_sims_post_cmbnncs"
        super().__init__(cfg, stage_str)
        # Cheating a bit with the stage string
        # TODO Better this
        # Have to refer to these differently
        # CMBNNCS has Pre- and Post-processing stages with outputs
        #   requiring imshow()
        self.in_cmb_map: Asset = self.assets_in["cmb_map_post"]

        self.right_subplot_title = "CMBNNCS Predicted"


class PetroffShowSimsPostExecutor(ShowSimsPostExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # Cheating a bit with the stage string
        # TODO Better this
        # Have to refer to these differently
        # Petroff has prediction requiring mollview()
        stage_str = "show_sims_pred_petroff"
        super().__init__(cfg, stage_str)
        self.in_cmb_map: Asset = self.assets_in["cmb_map_pred"]

        self.right_subplot_title = "Petroff Predicted"


class NILCShowSimsPostExecutor(ShowSimsPostExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # Cheating a bit with the stage string
        # TODO Better this
        # Have to refer to these differently
        # ILC methods have prediction requiring mollview()
        stage_str = "show_sims_pred_nilc"
        super().__init__(cfg, stage_str)
        self.in_cmb_map: Asset = self.assets_in["cmb_map_pred"]

        self.right_subplot_title = "NILC Predicted"
