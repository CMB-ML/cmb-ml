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
from cmbml.core.asset_handlers.asset_handlers_base import EmptyHandler, Config
from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap
from cmbml.core.asset_handlers.qtable_handler import QTableHandler  # Import to register handler

from cmbml.cmbnncs_local.handler_npymap import NumpyMap
from cmbml.utils.planck_instrument import make_instrument, Instrument
from cmbml.utils import planck_cmap


logger = logging.getLogger(__name__)


class ShowSimsExecutor(BaseStageExecutor):
    """
    Abstract.

    Makes pairs of images of maps - one is always the simulation CMB. The other
        depends on the child class.

    Will only make a subset of test set as images given by `override_n_sims`
        in the pipeline yaml for this stage.
    """
    def __init__(self, cfg: DictConfig, stage_str: str) -> None:
        # The following stage_str must match the pipeline yaml
        super().__init__(cfg, stage_str=stage_str)

        if self.__class__.__name__ == "ShowSimsExecutor":
            # TODO: Can I ABC this?
            raise NotImplementedError("This is a base class.")

        in_det_table: Asset  = self.assets_in['planck_deltabandpass']
        in_det_table_handler: QTableHandler
        det_info = in_det_table.read()
        self.instrument: Instrument = make_instrument(cfg=cfg, det_info=det_info)

        if self.override_sim_nums is None:
            logger.warning("No particular sim indices specified. Outputs will be produced for all. This is not recommended.")
        self.min_max = self.get_plot_min_max()
        self.fig_model_name = cfg.fig_model_name

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

    def make_maps_per_field(self, map_sim, map_mang, det, scale_factors, out_asset):
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

                self.make_mollview(map_sim[field_idx], ax1, show_cbar=True)

                scale_factor = scale_factors[field_str]['scale']
                unscaled_map_mang = map_mang[field_idx] * scale_factor
                self.make_imshow(unscaled_map_mang, ax2)

                norm = plt.Normalize(vmin=self.min_max[0], vmax=self.min_max[1])
                sm = plt.cm.ScalarMappable(cmap=planck_cmap.colombi1_cmap, norm=norm)
                sm.set_array([])

                map_unit = map_sim[field_idx].unit.to_string('latex_inline')

                cb = fig.colorbar(sm, cax=cbar_ax)
                cb.set_label(f'Intensity ({map_unit})')

                self.save_figure(title_start, split, sim_n, field_str, out_asset)

    def save_figure(self, title, split_name, sim_num, field_str, out_asset):
        plt.suptitle(f"{title}, {split_name}:{sim_num} {field_str} Stokes")

        fn = out_asset.path
        out_asset.write()
        plt.savefig(fn)
        plt.close()

    def make_imshow(self, some_map, ax):
        plt.axes(ax)
        plot_params = dict(
            vmin=self.min_max[0],
            vmax=self.min_max[1],
            cmap=planck_cmap.colombi1_cmap,
        )
        plt.imshow(some_map.value, **plot_params)
        plt.title(self.right_subplot_title)
        ax.set_axis_off()
        # cb = plt.colorbar()
        # map_unit = some_map.unit.to_string('latex_inline')
        # cb.set_label(f'Intensity ({map_unit})')

    def make_mollview(self, some_map, ax, min_or=None, max_or=None, show_cbar=False, unit=None, title="Raw Simulation"):
        if isinstance(some_map, u.Quantity):
            unit = some_map.unit.to_string('latex_inline')
            to_plot = some_map.value
        else:
            to_plot = some_map
        plt.axes(ax)
        vmin = self.min_max[0] if min_or is None else min_or
        vmax = self.min_max[1] if max_or is None else max_or
        plot_params = dict(
            xsize=2400,
            min=vmin, 
            max=vmax,
            unit=unit,
            cmap=planck_cmap.colombi1_cmap,
            hold=True,
            cbar=show_cbar
        )
        hp.mollview(to_plot, **plot_params)

        plt.title(title)


def load_sim_and_mang_map(sim_asset, mang_asset, cen_freq):
    sim_map = sim_asset.read().to(u.uK_CMB)
    if cen_freq == 'cmb':
        sim_map = sim_map.to(u.uK_CMB)
    else:
        sim_map = sim_map.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(cen_freq))
    mang_map = mang_asset.read() * u.uK_CMB
    return sim_map, mang_map


class ShowSimsPrepExecutor(ShowSimsExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        stage_str = "show_sims_prep_cmbnncs"
        super().__init__(cfg, stage_str)

        self.right_subplot_title = "Preprocessed"

        self.out_cmb_figure: Asset = self.assets_out["cmb_map_render"]
        self.out_obs_figure: Asset = self.assets_out["obs_map_render"]
        out_cmb_figure_handler: EmptyHandler
        out_obs_figure_handler: EmptyHandler

        self.in_cmb_map_sim: Asset = self.assets_in["cmb_map_sim"]
        self.in_cmb_map_prep: Asset = self.assets_in["cmb_map_prep"]
        self.in_obs_map_sim: Asset = self.assets_in["obs_maps_sim"]
        self.in_obs_map_prep: Asset = self.assets_in["obs_maps_prep"]
        self.in_norm_file: Asset = self.assets_in["norm_file"]
        in_cmb_map_handler: NumpyMap
        in_obs_map_handler: NumpyMap
        in_norm_file_handler: Config

    def process_sim(self) -> None:
        scale_factors = self.in_norm_file.read()
        cmb_map_sim, cmb_map_prep = load_sim_and_mang_map(self.in_cmb_map_sim, self.in_cmb_map_prep, 'cmb')
        self.make_maps_per_field(cmb_map_sim, 
                                 cmb_map_prep, 
                                 det="cmb", 
                                 scale_factors=scale_factors['cmb'],
                                 out_asset=self.out_cmb_figure)
        for freq, detector in self.instrument.dets.items():
            with self.name_tracker.set_context("freq", freq):
                obs_map_sim, obs_map_prep = load_sim_and_mang_map(self.in_obs_map_sim, 
                                                                  self.in_obs_map_prep, 
                                                                  detector.cen_freq)
                self.make_maps_per_field(obs_map_sim, 
                                         obs_map_prep, 
                                         det=freq, 
                                         scale_factors=scale_factors[freq],
                                         out_asset=self.out_obs_figure)


class CMBNNCSShowSimsPredExecutor(ShowSimsExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        stage_str = "show_cmb_pred_cmbnncs"
        super().__init__(cfg, stage_str)

        self.right_subplot_title = "Predicted"

        self.out_cmb_figure: Asset = self.assets_out["cmb_map_render"]
        out_cmb_figure_handler: EmptyHandler

        self.in_cmb_map_sim: Asset = self.assets_in["cmb_map_sim"]
        self.in_cmb_map_pred: Asset = self.assets_in["cmb_map_pred"]
        self.in_norm_file: Asset = self.assets_in["norm_file"]
        in_cmb_map_sim_handler: HealpyMap
        in_cmb_map_pred_handler: NumpyMap
        in_norm_file_handler: Config

    def process_sim(self) -> None:
        logger.debug(f"Reading norm_file from: {self.in_norm_file.path}")
        scale_factors = self.in_norm_file.read()
        for epoch in self.model_epochs:
            logger.info(f"Creating map figures predictions, model epoch {epoch}")
            with self.name_tracker.set_context('epoch', epoch):
                cmb_map_sim, cmb_map_pred = load_sim_and_mang_map(self.in_cmb_map_sim, 
                                                                  self.in_cmb_map_pred, 
                                                                  'cmb')
                self.make_maps_per_field(cmb_map_sim,
                                         cmb_map_pred,
                                         det="cmb",
                                         scale_factors=scale_factors['cmb'],
                                         out_asset=self.out_cmb_figure)


class ShowSimsPostExecutor(ShowSimsExecutor):
    def __init__(self, cfg: DictConfig, stage_str=None) -> None:
        stage_str = "show_cmb_post_masked"
        super().__init__(cfg, stage_str)

        self.suptitle = "CMB Predictions"
        self.right_subplot_title = "Predicted"

        self.out_cmb_figure: Asset = self.assets_out["cmb_map_render"]
        out_cmb_figure_handler: EmptyHandler

        self.in_cmb_map_post: Asset = self.assets_in["cmb_map_post"]
        self.in_cmb_map_sim: Asset = self.assets_in["cmb_map_sim"]
        in_cmb_map_handler: HealpyMap

    def process_sim(self) -> None:
        for epoch in self.model_epochs:
            with self.name_tracker.set_context('epoch', epoch):
                cmb_map_sim = self.in_cmb_map_sim.read()
                cmb_map_post = self.in_cmb_map_post.read()
                self.make_maps_per_field(cmb_map_sim, 
                                         cmb_map_post, 
                                         out_asset=self.out_cmb_figure)

    def make_maps_per_field(self, map_sim, map_post, out_asset):
        """
        Makes a figure for each field in the maps (e.g., IQU will result in 3 figures)
        """
        split = self.name_tracker.context['split']
        sim_n = f"{self.name_tracker.context['sim_num']:0{self.cfg.file_system.sim_str_num_digits}d}"
        fields = self.cfg.scenario.map_fields

        for field_str in fields:
            with self.name_tracker.set_context("field", field_str):
                field_idx = {'I': 0, 'Q': 1, 'U': 2}[field_str]
                fig = plt.figure(figsize=(30, 7), dpi=150)
                gs = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 6], wspace=0.1)

                axs = [plt.subplot(gs[i]) for i in range(4)]

                mask = map_sim[field_idx] == hp.UNSEEN

                diff = map_post[field_idx] - map_sim[field_idx]

                diff = hp.ma(diff)
                diff.mask = mask

                plot_params = dict(show_cbar=True, unit=map_post.unit.to_string('latex_inline'))

                self.make_mollview(map_sim[field_idx], axs[0], title="Realization", **plot_params)
                self.make_mollview(map_post[field_idx], axs[1], title="Prediction", **plot_params)
                self.make_mollview(diff, axs[2], title="Difference", min_or=-120, max_or=120, **plot_params)

                # healpy applies the graticule to every subplot, so we only need to do it once; 
                # if this were in self.make_mollview, earlier subplots will have multiple graticules applied.
                hp.graticule(dpar=45, dmer=45)

                n_bins = 50

                plt.axes(axs[3])
                plt.hist(diff.compressed(), bins=n_bins, range=(-120, 120), color="#524FA1", histtype='stepfilled')
                axs[3].set_yticks([])
                axs[3].set_xlabel("Deviation from Zero Difference ($\\mu \\text{K}_\\text{CMB}$)")
                axs[3].set_ylabel("Pixel Count")
                axs[3].set_title("Histogram of Difference")
                for x in [-100, -50, 0, 50, 100]:
                    axs[3].axvline(x=x, color='black', linestyle='--', linewidth=0.5)
                self.save_figure(self.suptitle, split, sim_n, field_str, out_asset)


class CMBNNCSShowSimsPostExecutor(ShowSimsPostExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.right_subplot_title = "CMBNNCS Predicted"


class PetroffShowSimsPostExecutor(ShowSimsPostExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.right_subplot_title = "Petroff Predicted"


class NILCShowSimsPostExecutor(ShowSimsPostExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.right_subplot_title = "NILC Predicted"


class CommonRealPostExecutor(ShowSimsPostExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        stage_str = "show_cmb_post_masked"
        super().__init__(cfg, stage_str=stage_str)


class CommonCMBNNCSShowSimsPostExecutor(CommonRealPostExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.right_subplot_title = "CMBNNCS Predicted"


class CommonPetroffShowSimsPostExecutor(CommonRealPostExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.right_subplot_title = "Petroff Predicted"


class CommonNILCShowSimsPostExecutor(CommonRealPostExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.right_subplot_title = "NILC Predicted"
        self.suptitle = cfg.fig_model_name
