import healpy as hp
import logging
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pysm3.units as u

from omegaconf import DictConfig, ListConfig

from cmbml.core import BaseStageExecutor
from cmbml.core.asset import Asset
from cmbml.core.asset_handlers import (
    EmptyHandler,
    NumpyMap,
    QTableHandler,
    Config
)
from cmbml.utils import make_instrument, Instrument
from cmbml.core.split import Split
from cmbml.utils import planck_cmap


logger = logging.getLogger(__name__)


class ShowSimsPrepExecutor(BaseStageExecutor):
    """
    Displays Molleweide and Mangled projections.
    
    Will only make a subset of test set as images given by `override_n_sims`
        in the pipeline yaml for this stage.
    """
    def __init__(self, cfg: DictConfig) -> None:
        # The following stage_str must match the pipeline yaml
        stage_str = "show_sims_prep"
        super().__init__(cfg, stage_str=stage_str)

        self.right_subplot_title = "Preprocessed"

        self.out_cmb_figure: Asset = self.assets_out["cmb_map_render"]
        self.out_obs_figure: Asset = self.assets_out["obs_map_render"]
        out_cmb_figure_handler: EmptyHandler
        out_obs_figure_handler: EmptyHandler

        in_det_table: Asset  = self.assets_in['deltabandpass']
        self.in_cmb_map_sim: Asset = self.assets_in["cmb_map_sim"]
        self.in_cmb_map_prep: Asset = self.assets_in["cmb_map_prep"]
        self.in_obs_map_sim: Asset = self.assets_in["obs_maps_sim"]
        self.in_obs_map_prep: Asset = self.assets_in["obs_maps_prep"]
        # self.in_dataset_stats: Asset = self.assets_in["dataset_stats"]
        in_det_table_handler: QTableHandler
        in_cmb_map_handler: NumpyMap
        in_obs_map_handler: NumpyMap
        in_dataset_stats_handler: Config

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

    def process_sim(self) -> None:
        # scale_factors = self.in_dataset_stats.read()
        cmb_map_sim, cmb_map_prep = self.load_sim_and_mang_map(self.in_cmb_map_sim, 
                                                               self.in_cmb_map_prep, 
                                                               'cmb')
        self.make_maps_per_field(cmb_map_sim, 
                                 cmb_map_prep, 
                                 det="cmb", 
                                #  scale_factors=scale_factors['cmb'],
                                 out_asset=self.out_cmb_figure)
        for freq, detector in self.instrument.dets.items():
            with self.name_tracker.set_context("freq", freq):
                obs_map_sim, obs_map_prep = self.load_sim_and_mang_map(self.in_obs_map_sim, 
                                                                  self.in_obs_map_prep, 
                                                                  detector.cen_freq)
                self.make_maps_per_field(obs_map_sim, 
                                         obs_map_prep, 
                                         det=freq, 
                                        #  scale_factors=scale_factors[freq],
                                         out_asset=self.out_obs_figure)

    def make_maps_per_field(self, 
                            map_sim, 
                            map_mang, 
                            det, 
                            # scale_factors, 
                            out_asset
                            ):
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

                # scale_factor = scale_factors[field_str]['scale']
                unscaled_map_mang = map_mang[field_idx] * 1
                # unscaled_map_mang = map_mang[field_idx] * scale_factor
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

    @staticmethod
    def load_sim_and_mang_map(sim_asset, mang_asset, cen_freq):
        sim_map = sim_asset.read().to(u.uK_CMB)
        if cen_freq == 'cmb':
            sim_map = sim_map.to(u.uK_CMB)
        else:
            sim_map = sim_map.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(cen_freq))
        mang_map = mang_asset.read() * u.uK_CMB
        return sim_map, mang_map

# class ShowSimsPrepExecutor(ShowSimsExecutor):
#     def __init__(self, cfg: DictConfig) -> None:
#         stage_str = "show_sims_prep"
#         super().__init__(cfg, stage_str)

#         self.right_subplot_title = "Preprocessed"

#         self.out_cmb_figure: Asset = self.assets_out["cmb_map_render"]
#         self.out_obs_figure: Asset = self.assets_out["obs_map_render"]
#         out_cmb_figure_handler: EmptyHandler
#         out_obs_figure_handler: EmptyHandler

#         self.in_cmb_map_sim: Asset = self.assets_in["cmb_map_sim"]
#         self.in_cmb_map_prep: Asset = self.assets_in["cmb_map_prep"]
#         self.in_obs_map_sim: Asset = self.assets_in["obs_maps_sim"]
#         self.in_obs_map_prep: Asset = self.assets_in["obs_maps_prep"]
#         # self.in_dataset_stats: Asset = self.assets_in["dataset_stats"]
#         in_cmb_map_handler: NumpyMap
#         in_obs_map_handler: NumpyMap
#         in_dataset_stats_handler: Config

    # def process_sim(self) -> None:
    #     # scale_factors = self.in_dataset_stats.read()
    #     cmb_map_sim, cmb_map_prep = self.load_sim_and_mang_map(self.in_cmb_map_sim, 
    #                                                            self.in_cmb_map_prep, 
    #                                                            'cmb')
    #     self.make_maps_per_field(cmb_map_sim, 
    #                              cmb_map_prep, 
    #                              det="cmb", 
    #                             #  scale_factors=scale_factors['cmb'],
    #                              out_asset=self.out_cmb_figure)
    #     for freq, detector in self.instrument.dets.items():
    #         with self.name_tracker.set_context("freq", freq):
    #             obs_map_sim, obs_map_prep = self.load_sim_and_mang_map(self.in_obs_map_sim, 
    #                                                               self.in_obs_map_prep, 
    #                                                               detector.cen_freq)
    #             self.make_maps_per_field(obs_map_sim, 
    #                                      obs_map_prep, 
    #                                      det=freq, 
    #                                     #  scale_factors=scale_factors[freq],
    #                                      out_asset=self.out_obs_figure)

