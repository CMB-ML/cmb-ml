"""Four-panel CMB postprocessing diagnostics for each simulation and beam."""

import logging

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pysm3.units as u
from omegaconf import DictConfig, ListConfig

from cmbml.core import BaseStageExecutor, Split, Asset
from cmbml.core.asset_handlers import MPLFigure
from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap  # Register handler
from cmbml.utils.beam_naming import name_beam_fwhms
from cmbml.utils import planck_cmap

logger = logging.getLogger(__name__)


def _field_values(cmb_map, index: int):
    """Return numerical HEALPix values for an I/Q/U field or scalar I map."""
    values = cmb_map.value if isinstance(cmb_map, u.Quantity) else cmb_map
    values = np.asarray(values)
    if values.ndim == 1:
        if index != 0:
            raise ValueError("Scalar input map only contains the I field")
        return values
    return values[index]


from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def make_cmb_post_figure(
    realization,
    prediction,
    mask,
    *,
    min_max=(-500, 500),
    diff_limits=None,
    unit=None,
    title=None,
    figsize=(24, 6),
    dpi=150,
):
    """Build realization, prediction, masked residual, and histogram panels."""

    real = np.asarray(realization)
    pred = np.asarray(prediction)
    keep = np.asarray(mask, dtype=bool)

    if real.shape != pred.shape or real.shape != keep.shape:
        raise ValueError(
            f"Map/mask shapes differ: "
            f"real={real.shape}, pred={pred.shape}, mask={keep.shape}"
        )

    diff = hp.ma(pred - real)
    diff.mask = ~keep

    if diff_limits is None:
        delta_max = np.percentile(np.abs(diff.compressed()), 99.99)
        diff_limits = (-delta_max, delta_max)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    cmap = planck_cmap.colombi1_cmap

    # Explicit panel positions in normalized figure coordinates.
    # (left, bottom, width, height)
    map_positions = [
        (0.02, 0.22, 0.225, 0.65),
        (0.265, 0.22, 0.225, 0.65),
        (0.51, 0.22, 0.225, 0.65),
    ]

    hist_position = (0.78, 0.22, 0.195, 0.65)

    map_titles = ("Realization", "Prediction", "Difference")
    display_maps = (real, pred, diff)

    for i, (map_data, label) in enumerate(zip(display_maps, map_titles)):
        limits = diff_limits if i == 2 else min_max

        # Create an axis occupying exactly the desired position.
        ax = fig.add_axes(map_positions[i])
        plt.sca(ax)

        # Healpy replaces the active axes, preserving their position.
        hp.mollview(
            map_data,
            fig=fig.number,
            hold=True,
            title="",
            min=limits[0],
            max=limits[1],
            cmap=cmap,
            cbar=False,
            xsize=2400,
        )

        ax = plt.gca()
        ax.set_title(label, fontsize=12, pad=10)

        # Explicit colorbar below each map.
        left, bottom, width, height = map_positions[i]

        cax = fig.add_axes([
            left + width * 0.15,
            0.12,
            width * 0.70,
            0.025,
        ])

        norm = Normalize(vmin=limits[0], vmax=limits[1])

        if limits[0] is None or limits[1] is None:
            norm.autoscale_None(np.asarray(map_data).ravel())

        sm = ScalarMappable(norm=norm, cmap=cmap)

        cb = fig.colorbar(
            sm,
            cax=cax,
            orientation="horizontal",
        )

        if unit:
            cb.set_label(unit, fontsize=10)

    # Histogram
    ax = fig.add_axes(hist_position)

    ax.hist(
        diff.compressed(),
        bins=50,
        range=diff_limits,
        color="#524FA1",
        histtype="stepfilled",
    )

    ax.set_xlabel(
        r"Deviation from zero difference ($\mu\mathrm{K}_{\mathrm{CMB}}$)"
    )
    ax.set_ylabel("Pixel count")
    ax.set_title("Histogram of difference", fontsize=12)

    # Use Matplotlib's automatically selected major ticks
    ax.set_xlim(*diff_limits)
    axv_locs = ax.get_xticks()

    for value in axv_locs:
        if diff_limits[0] <= value <= diff_limits[1]:
            ax.axvline(
                value,
                color="black",
                linestyle="--",
                linewidth=0.5,
            )

    if title:
        fig.suptitle(title, fontsize=14, y=0.98)

    return fig


class ShowSimsPostExecutor(BaseStageExecutor):
    """Render one four-panel figure per split/simulation/epoch/beam/field."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="show_cmb_post_masked")

        self.out_cmb_figure: Asset = self.assets_out["cmb_map_render"]
        out_cmb_figure_handler: MPLFigure

        self.in_cmb_map_post: Asset = self.assets_in["cmb_map_post"]
        self.in_cmb_map_sim: Asset = self.assets_in["cmb_map_sim"]
        self.in_mask: Asset = self.assets_in["mask"]
        in_cmb_map_handler: HealpyMap
        in_mask_handler: HealpyMap

        self.target_beams = name_beam_fwhms(cfg.model.analysis.target_beam_fwhm)
        self.fig_type = cfg.get("fig_type", None)
        self.plot_min_max = self.get_stage_element("plot_min_max")
        self.mask = None

    def execute(self) -> None:
        self.out_cmb_figure.handler.set_fig_types(self.fig_type)
        # The sharp mask is already binary and at the correct NSIDE.
        self.mask = np.asarray(
            self.in_mask.read(map_fields=self.in_mask.use_fields)[0], dtype=bool
        )
        self.default_execute()

    def process_split(self, split: Split) -> None:
        sims = split.iter_sims() if self.override_sim_nums is None else self.override_sim_nums
        for sim in sims:
            with self.name_tracker.set_context("sim_num", sim):
                self.process_sim()

    def process_sim(self) -> None:
        epochs = self.model_epochs if self.model_epochs else [""]
        for epoch in epochs:
            with self.name_tracker.set_context("epoch", epoch):
                for beam_name, _ in self.target_beams:
                    with self.name_tracker.set_context("beam_name", beam_name):
                        real_map = self.in_cmb_map_sim.read()
                        pred_map = self.in_cmb_map_post.read()
                        self.make_maps_per_field(real_map, pred_map)

    def make_maps_per_field(self, real_map, pred_map) -> None:
        value = self.plot_min_max
        if value is None:
            limits = (None, None)
        elif isinstance(value, (int, float)):
            limits = (-value, value)
        elif isinstance(value, (list, tuple, ListConfig)) and len(value) == 2:
            limits = tuple(value)
        else:
            raise ValueError(f"Invalid plot_min_max: {value!r}")

        unit = pred_map.unit.to_string("latex_inline") if isinstance(pred_map, u.Quantity) else None
        for field in self.map_fields:
            index = {"I": 0, "Q": 1, "U": 2}[field]
            with self.name_tracker.set_context("field", field):
                fig = make_cmb_post_figure(
                    _field_values(real_map, index),
                    _field_values(pred_map, index),
                    self.mask,
                    min_max=limits,
                    unit=unit,
                    title=f"CMB predictions — {field}",
                )
                try:
                    self.out_cmb_figure.write(fig=fig)
                finally:
                    plt.close(fig)
