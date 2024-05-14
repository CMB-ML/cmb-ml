import logging

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from omegaconf import DictConfig

from core import (
    BaseStageExecutor,
    Asset,
    GenericHandler,
    )
from core.asset_handlers import Mover, Config, EmptyHandler

logger = logging.getLogger(__name__)


class PixelSummaryExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        logger.debug("Initializing PixelSummaryExecutor")
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="pixel_summary")

        self.in_report: Asset = self.assets_in["report"]
        in_report_handler: Config

        self.out_overall_stats: Asset = self.assets_out["overall_stats"]
        self.out_stats_per_split: Asset = self.assets_out["stats_per_split"]
        self.out_boxplots: Asset = self.assets_out["boxplots"]
        self.out_histogram: Asset = self.assets_out["histogram"]
        out_overall_stats_handler: EmptyHandler
        out_stats_per_split_handler: EmptyHandler
        out_boxplots_handler: EmptyHandler
        out_histogram_handler: EmptyHandler

        self.labels_lookup = self.get_labels_lookup()

    def get_labels_lookup(self):
        lookup = dict(self.cfg.model.analysis.px_functions)
        return lookup

    def execute(self) -> None:
        logger.debug("PixelSummaryExecutor execute() method.")
        report_contents = self.in_report.read()
        df = pd.DataFrame(report_contents)

        df = self.sort_order(df)
        df['epoch'] = df['epoch'].astype(str)

        for epoch in self.model_epochs:
            logger.info(f"Generating summary for epoch {epoch}.")
            with self.name_tracker.set_context('epoch', epoch):
                epoch_df = df[df['epoch']==str(epoch)]
                self.summary_tables(epoch_df)
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

    def summary_tables(self, df):
        # Compute overall averages, excluding non-numeric fields like 'sim' and 'split'
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        overall_stats = df[numeric_columns].agg(['mean', 'std'])
        stats_per_split = df.groupby('split')[numeric_columns].agg(['mean', 'std'])

        # Every now and then, it's ok to just comment out code. I promise this to you, Jim. - Jim
        # print(f"For epoch {self.name_tracker.context['epoch']}")
        # print("Overall:")
        # print(overall_stats)
        # print("\n per Split:")
        # print(stats_per_split)
        overall_path = self.out_overall_stats.path
        self.out_overall_stats.write()
        overall_stats.reset_index().to_csv(overall_path, index=False)
        per_split_path = self.out_stats_per_split.path
        stats_per_split.reset_index().to_csv(per_split_path, index=False)

    def make_split_histograms(self, df):
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        for metric in numeric_columns:
            with self.name_tracker.set_context("metric", metric):
                if "{metric}" not in self.out_histogram.path_template:
                    logger.warning("If multiple metrics were used, they will be overwritten by the last one. #WAAH")
                path = self.out_histogram.path
                g = sns.FacetGrid(df, col="split", col_wrap=4, height=3)
                g.map(sns.histplot, metric)
                g.figure.suptitle(f"Distribution of {metric} by Split, Epoch {self.name_tracker.context['epoch']}")
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

        plt.suptitle(f"Metric Distribution by Split, Epoch {self.name_tracker.context['epoch']}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        path = self.out_boxplots.path
        plt.savefig(path)
        plt.close()
