import logging

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from omegaconf import DictConfig

from src.core import (
    BaseStageExecutor,
    Asset,
    GenericHandler,
    )
from src.core.asset_handlers.asset_handlers_base import EmptyHandler # Import for typing hint
from src.core.asset_handlers.asset_handlers_base import Config # Import for typing hint

logger = logging.getLogger(__name__)


class PixelSummaryFigsExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="pixel_summary_figs")

        self.in_report: Asset = self.assets_in["report"]
        in_report_handler: Config

        self.out_boxplots: Asset = self.assets_out["boxplots"]
        self.out_histogram: Asset = self.assets_out["histogram"]
        out_boxplots_handler: EmptyHandler
        out_histogram_handler: EmptyHandler

        self.labels_lookup = self.get_labels_lookup()
        self.fig_model_name = cfg.fig_model_name

    def get_labels_lookup(self):
        lookup = dict(self.cfg.model.analysis.px_functions)
        return lookup

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute().")
        report_contents = self.in_report.read()
        df = pd.DataFrame(report_contents)

        df = self.sort_order(df)
        df['epoch'] = df['epoch'].astype(str)

        for epoch in self.model_epochs:
            logger.info(f"Generating summary for epoch {epoch}.")
            with self.name_tracker.set_context('epoch', epoch):
                epoch_df = df[df['epoch']==str(epoch)]
                self.make_split_histograms(epoch_df)
                self.make_boxplots(epoch_df)

    def sort_order(self, df):
        # Sort Split Order for tables and figures
        try:
            # The default order of splits is lexicographic; putting Test10 between Test1 and Test2
            split_order = sorted(df['split'].unique(), key=lambda x: int(x.replace('Test', '')))
            # Convert 'split' to a categorical type with the defined order
            df['split'] = pd.Categorical(df['split'], categories=split_order, ordered=True)
        except:
            # Failure is ok. An ugly order can be sorted out later.
            pass

        return df

    def make_split_histograms(self, df):
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        for metric in numeric_columns:
            with self.name_tracker.set_context("metric", metric):
                if "{metric}" not in self.out_histogram.path_template:
                    logger.warning("If multiple metrics were used, they will be overwritten by the last one.")
                self.out_histogram.write()  # Make directory
                path = self.out_histogram.path
                g = sns.FacetGrid(df, col="split", col_wrap=4, height=3)
                g.map(sns.histplot, metric)
                g.figure.suptitle(f"")
                plt.subplots_adjust(top=0.9)                             # Adjust the top margin to fit the suptitle
                g.savefig(path)
                plt.close(g.figure)

    def make_boxplots(self, df):
        metrics = list(self.labels_lookup.keys())
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

        for ax, metric in zip(axes.flatten(), metrics):
            df.boxplot(column=metric, by='split', ax=ax)
            ax.set_title(self.labels_lookup[metric]["plot_name"])
            ax.set_ylabel(self.labels_lookup[metric]["axis_name"])

        title = self.make_title(metric)
        plt.suptitle(title)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.out_boxplots.write()  # Make directory
        path = self.out_boxplots.path
        plt.savefig(path)
        plt.close()

    def make_title(self, metric):
        return f"{self.fig_model_name} Per Pixel Distribution of {metric} by Split, Epoch {self.name_tracker.context['epoch']}"