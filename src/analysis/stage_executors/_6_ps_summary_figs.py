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


class PowerSpectrumSummaryFigsExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="ps_summary_figs")

        self.in_report: Asset = self.assets_in["report"]
        in_report_handler: Config

        self.out_boxplots: Asset = self.assets_out["boxplots"]
        self.out_histogram: Asset = self.assets_out["histogram"]
        out_boxplots_handler: EmptyHandler
        out_histogram_handler: EmptyHandler

        self.labels_lookup = self.get_labels_lookup()
        self.fig_model_name = cfg.fig_model_name

    def get_labels_lookup(self):
        lookup = dict(self.cfg.model.analysis.ps_functions)
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
                if epoch_df.empty:  # For models without epochs (ILC)
                    epoch_df = df
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
        baseline_groups = {name: group for name, group in df.groupby('baseline')}
        for baseline, group_df in baseline_groups.items():
            for metric in numeric_columns:
                context_dict = dict(metric=metric, baseline=baseline)
                with self.name_tracker.set_contexts(context_dict):
                    if "{metric}" not in self.out_histogram.path_template:
                        logger.warning("If multiple metrics were used, they will be overwritten by the last one.")
                    self.out_histogram.write()
                    path = self.out_histogram.path
                    g = sns.FacetGrid(group_df, col="split", col_wrap=4, height=3)
                    g.map(sns.histplot, metric)
                    title = self.make_title(epoch=self.name_tracker.context['epoch'],
                                            baseline=baseline,
                                            metric=metric,
                                            n_splits=group_df['split'].nunique())
                    g.figure.suptitle(title)
                    plt.subplots_adjust(top=0.9)                             # Adjust the top margin to fit the suptitle
                    g.savefig(path)
                    plt.close(g.figure)

    def make_boxplots(self, df):
        metrics = list(self.labels_lookup.keys())
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

        baseline_groups = {name: group for name, group in df.groupby('baseline')}
        for baseline, group_df in baseline_groups.items():
            context_dict = dict(baseline=baseline)
            with self.name_tracker.set_contexts(context_dict):
                for ax, metric in zip(axes.flatten(), metrics):
                    group_df.boxplot(column=metric, by='split', ax=ax)
                    ax.set_title(self.labels_lookup[metric]["plot_name"])
                    ax.set_ylabel(self.labels_lookup[metric]["axis_name"])
                epoch = self.name_tracker.context['epoch']
                title = self.make_title(epoch=epoch, baseline=baseline, n_splits=group_df['split'].nunique())
                plt.suptitle(title)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                self.out_boxplots.write()
                path = self.out_boxplots.path
                plt.savefig(path)
                plt.close()

    def make_title(self, epoch, baseline, n_splits, metric=None):
        if epoch != '':
            e_phrase = f", Epoch {epoch}"
        else:
            e_phrase = ""
        if baseline == "real":
            b_phrase = " Relative to Realization"
        elif baseline == "thry":
            b_phrase = " Relative to Theory"
        else:
            b_phrase = ""
        if metric is not None:
            m_phrase = f" of {metric}"
        else:
            m_phrase = ""
        if n_splits == 1:
            s_phrase = ""
        else:
            s_phrase = " by Split"
        title = f"{self.fig_model_name} Power Spectra{b_phrase} Distribution{m_phrase}{s_phrase}{e_phrase}"
        return title

