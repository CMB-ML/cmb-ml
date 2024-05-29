import logging

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from omegaconf import DictConfig

from src.core import BaseStageExecutor, Asset
from src.core.asset_handlers.asset_handlers_base import Config      # Import for typing hint
from src.core.asset_handlers.pd_csv_handler import PandasCsvHandler # Import for typing hint

logger = logging.getLogger(__name__)


class PixelSummaryExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="pixel_summary_tables")

        self.out_overall_stats: Asset = self.assets_out["overall_stats"]
        self.out_stats_per_split: Asset = self.assets_out["stats_per_split"]
        out_overall_stats_handler: PandasCsvHandler
        out_stats_per_split_handler: PandasCsvHandler

        self.in_report: Asset = self.assets_in["report"]
        in_report_handler: Config

        self.labels_lookup = self.get_labels_lookup()

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
                self.make_summary_tables(epoch_df)

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

    def make_summary_tables(self, df):
        # Compute overall averages, excluding non-numeric fields like 'split'
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        # Also get rid of the 'sim' column, ('sim' is just an ID number, no statistics wanted.)
        numeric_columns = numeric_columns.drop('sim')

        overall_stats = df[numeric_columns].agg(['mean', 'std'])
        overall_stats.reset_index()
        self.out_overall_stats.write(data=overall_stats, index=True)

        stats_per_split = df.groupby('split')[numeric_columns].agg(['mean', 'std'])
        stats_per_split.reset_index()
        self.out_stats_per_split.write(data=stats_per_split, index=True)
