"""Summarize pixel-space metrics across simulations, splits, and beams."""

import logging
import re

import pandas as pd
from omegaconf import DictConfig

from cmbml.core import Asset, BaseStageExecutor
from cmbml.core.asset_handlers import Config
from cmbml.core.asset_handlers.pd_csv_handler import PandasCsvHandler  # Register handler
from cmbml.utils.beam_naming import name_beam_fwhms

logger = logging.getLogger(__name__)


def _split_sort_key(name: str) -> tuple[int, int, str]:
    """Order Test, Test0, Test1, ..., then other split names."""
    match = re.fullmatch(r"Test(\d*)", str(name))
    if match:
        return (0, int(match.group(1)) if match.group(1) else -1, "")
    return (1, 0, str(name))


class PixelSummaryExecutor(BaseStageExecutor):
    """Write overall and per-split metric summaries for each epoch.

    Overall rows are indexed by beam_name; per-split rows by split and
    beam_name. Both tables have (metric, mean/std) columns.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="pixel_summary_tables")

        self.out_overall_stats: Asset = self.assets_out["overall_stats"]
        self.out_stats_per_split: Asset = self.assets_out["stats_per_split"]
        out_overall_stats_handler: PandasCsvHandler
        out_stats_per_split_handler: PandasCsvHandler

        self.in_report: Asset = self.assets_in["report"]
        in_report_handler: Config

        self.metric_names = list(cfg.model.analysis.px_functions.keys())
        self.beam_names = [
            name for name, _ in name_beam_fwhms(cfg.model.analysis.target_beam_fwhm)
        ]

    def execute(self) -> None:
        report = self.in_report.read()
        df = pd.DataFrame(report)
        required = {"split", "sim", "epoch", "beam_name", *self.metric_names}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Pixel report missing columns: {sorted(missing)}")
        if df.empty:
            raise ValueError("Pixel report contains no records")

        df["epoch"] = df["epoch"].astype(str)
        df["split"] = pd.Categorical(
            df["split"],
            categories=sorted(df["split"].unique(), key=_split_sort_key),
            ordered=True,
        )
        df["beam_name"] = pd.Categorical(
            df["beam_name"], categories=self.beam_names, ordered=True
        )
        if df["beam_name"].isna().any():
            raise ValueError("Pixel report includes beam names not in target_beam_fwhm")

        for epoch in self.model_epochs:
            epoch_str = str(epoch)
            epoch_df = df.loc[df["epoch"] == epoch_str]
            if epoch_df.empty:
                logger.warning("No pixel-analysis records for epoch %s; skipping", epoch_str)
                continue
            logger.info("Generating pixel summary for epoch %s", epoch_str)
            with self.name_tracker.set_context("epoch", epoch):
                self.make_summary_tables(epoch_df)

    def make_summary_tables(self, df: pd.DataFrame) -> None:
        # observed=True prevents emitting rows for beams/splits with no data.
        overall = df.groupby("beam_name", observed=True)[self.metric_names].agg(
            ["mean", "std"]
        )
        per_split = df.groupby(
            ["split", "beam_name"], observed=True, sort=True
        )[self.metric_names].agg(["mean", "std"])

        # Preserve the CSV handlers' existing convention: index columns are written.
        self.out_overall_stats.write(data=overall, index=True)
        self.out_stats_per_split.write(data=per_split, index=True)
