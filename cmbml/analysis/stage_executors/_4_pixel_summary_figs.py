"""Plot distributions of per-simulation pixel statistics, separately by beam."""

import logging
import math
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from cmbml.core import Asset, BaseStageExecutor
from cmbml.core.asset_handlers import Config, MPLFigure
from cmbml.utils.beam_naming import name_beam_fwhms

logger = logging.getLogger(__name__)


def _split_sort_key(name: str) -> tuple[int, int, str]:
    """Place Test, Test0, Test1, ... before other named splits."""
    match = re.fullmatch(r"Test(\d*)", str(name))
    if match:
        return (0, int(match.group(1)) if match.group(1) else -1, "")
    return (1, 0, str(name))


def make_pixel_boxplots_figure(
    df: pd.DataFrame,
    *,
    metric_labels: dict,
    split_order: list[str],
    title: str,
):
    """Create a metric grid of Matplotlib boxplots comparing splits at one beam."""
    metric_names = list(metric_labels)
    ncols = 3
    nrows = math.ceil(len(metric_names) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 4.0 * nrows), squeeze=False
    )

    for ax, metric in zip(axes.flat, metric_names):
        data_by_split = [
            pd.to_numeric(
                df.loc[df["split"] == split, metric], errors="coerce"
            ).dropna().to_numpy()
            for split in split_order
        ]

        # Leave missing splits out rather than passing empty arrays to boxplot.
        present = [(split, values) for split, values in zip(split_order, data_by_split)
                   if len(values)]
        if present:
            ax.boxplot(
                [values for _, values in present],
                tick_labels=[split for split, _ in present],
                patch_artist=True,
                boxprops={"facecolor": "steelblue", "alpha": 0.6},
            )

        metadata = metric_labels[metric]
        ax.set_title(metadata.get("plot_name", metric))
        ax.set_xlabel("Split")
        ax.set_ylabel(metadata.get("axis_name", metric))
        ax.tick_params(axis="x", rotation=45)

    for ax in list(axes.flat)[len(metric_names):]:
        ax.set_visible(False)

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def make_pixel_histogram_figure(
    df: pd.DataFrame,
    *,
    metric: str,
    metric_label: str,
    split_order: list[str],
    title: str,
    bins: int = 30,
):
    """Create histograms with shared bin edges and axes across splits."""
    ncols = min(4, len(split_order))
    nrows = math.ceil(len(split_order) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows),
        squeeze=False, sharex=True, sharey=True,
    )
    values = pd.to_numeric(df[metric], errors="coerce").dropna()
    values = values[np.isfinite(values)]
    if values.empty:
        raise ValueError(f"No finite numeric values for metric {metric!r}")

    edges = np.histogram_bin_edges(values.to_numpy(), bins=bins)

    for ax, split in zip(axes.flat, split_order):
        subset = pd.to_numeric(
            df.loc[df["split"] == split, metric], errors="coerce"
        ).dropna()
        subset = subset[np.isfinite(subset)]
        ax.hist(subset.to_numpy(), bins=edges, color="#524FA1")
        ax.set_title(f"{split} (n={len(subset)})")
        ax.set_xlabel(metric_label)
        ax.set_ylabel("Simulations")

    for ax in list(axes.flat)[len(split_order):]:
        ax.set_visible(False)

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


class PixelSummaryFigsExecutor(BaseStageExecutor):
    """Write beam-separated boxplots and split-faceted metric histograms."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="pixel_summary_figs")

        self.in_report: Asset = self.assets_in["report"]
        in_report_handler: Config

        self.out_boxplots: Asset = self.assets_out["boxplots"]
        self.out_histogram: Asset = self.assets_out["histogram"]
        out_boxplots_handler: MPLFigure
        out_histogram_handler: MPLFigure

        self.metric_labels = dict(cfg.model.analysis.px_functions)
        self.metric_names = list(self.metric_labels)
        self.beam_names = [
            name for name, _ in name_beam_fwhms(cfg.model.analysis.target_beam_fwhm)
        ]
        self.fig_model_name = cfg.fig_model_name
        self.fig_type = cfg.get("fig_type", None)

    def execute(self) -> None:
        self.out_boxplots.handler.set_fig_types(self.fig_type)
        self.out_histogram.handler.set_fig_types(self.fig_type)

        df = pd.DataFrame(self.in_report.read())
        required = {"split", "sim", "epoch", "beam_name", *self.metric_names}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Pixel report missing columns: {sorted(missing)}")
        if df.empty:
            raise ValueError("Pixel report contains no records")

        df["epoch"] = df["epoch"].astype(str)
        unknown = set(df["beam_name"].dropna()) - set(self.beam_names)
        if unknown:
            raise ValueError(f"Pixel report contains unknown beams: {sorted(unknown)}")

        for epoch in self.model_epochs:
            epoch_df = df.loc[df["epoch"] == str(epoch)]
            if epoch_df.empty:
                logger.warning("No pixel report entries for epoch %s", epoch)
                continue
            with self.name_tracker.set_context("epoch", epoch):
                for beam_name in self.beam_names:
                    beam_df = epoch_df.loc[epoch_df["beam_name"] == beam_name]
                    if beam_df.empty:
                        logger.warning("No data for epoch %s, beam %s", epoch, beam_name)
                        continue
                    with self.name_tracker.set_context("beam_name", beam_name):
                        self.make_figures(beam_df, epoch=epoch, beam_name=beam_name)

    def make_figures(self, df: pd.DataFrame, *, epoch, beam_name: str) -> None:
        split_order = sorted(df["split"].unique(), key=_split_sort_key)
        title_prefix = f"{self.fig_model_name} — {beam_name} — epoch {epoch}"

        fig = make_pixel_boxplots_figure(
            df, metric_labels=self.metric_labels,
            split_order=split_order,
            title=f"{title_prefix}: pixel statistics by split",
        )
        try:
            self.out_boxplots.write(fig=fig)
        finally:
            plt.close(fig)

        for metric in self.metric_names:
            with self.name_tracker.set_context("metric", metric):
                fig = make_pixel_histogram_figure(
                    df,
                    metric=metric,
                    metric_label=self.metric_labels[metric].get("axis_name", metric),
                    split_order=split_order,
                    title=f"{title_prefix}: {self.metric_labels[metric].get('plot_name', metric)}",
                )
                try:
                    self.out_histogram.write(fig=fig)
                finally:
                    plt.close(fig)
