import logging
from typing import Union, Tuple, List, Any, Dict

from omegaconf import DictConfig

import numpy as np
import pandas as pd
import json

from cmbml.core import (BaseStageExecutor,
                       Split, 
                       Asset)

from cmbml.core.asset_handlers.asset_handlers_base import EmptyHandler
from cmbml.core.asset_handlers.ps_handler import NumpyPowerSpectrum
from cmbml.core.asset_handlers.pd_csv_handler import PandasCsvHandler

logger = logging.getLogger(__name__)

class PowerSpectrumSummaryExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="ps_summary_tables")

        self.epoch_stats: Asset = self.assets_out["epoch_stats"]
        out_ps_stats_handlers: PandasCsvHandler

        self.in_ps_report: Asset = self.assets_in["report"]
        in_ps_report_handler: PandasCsvHandler

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        df = self.in_ps_report.read()
        # print(df.info())
        df['epoch'] = df['epoch'].astype(str)
        df['epoch'] = df['epoch'].replace("nan", "")

        rel_to = df['baseline'].unique()

        all_summaries = []
        for baseline in rel_to:
            rel_to_df = df[df['baseline'] == baseline]
            for epoch in self.model_epochs:
                logger.info(f"Generating summary for epoch {epoch}, relative to {baseline}.")
                context_dict = dict(epoch=epoch, baseline=baseline)
                with self.name_tracker.set_contexts(context_dict):
                    epoch_df = rel_to_df[rel_to_df['epoch']==str(epoch)]
                    summary_df = self.summary_tables(epoch_df)
                    summary_df['epoch'] = epoch  # Add epoch as a column
                    summary_df['baseline'] = baseline  # Add baseline as a column
                    all_summaries.append(summary_df)
        final_summary = pd.concat(all_summaries, ignore_index=True)
        self.epoch_stats.write(data=final_summary, index=False)

    def summary_tables(self, df):
        # Compute overall averages, excluding non-numeric fields like 'sim' and 'split'
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        numeric_columns = numeric_columns.drop('sim')

        # Get mean and st dev
        epoch_stats = df[numeric_columns].agg(['mean', 'std'])
        epoch_stats = epoch_stats.unstack().reset_index()
        epoch_stats.columns = ['metric', 'type', 'value']
        return epoch_stats
        # self.epoch_stats.write(data=epoch_stats, index=True)

        # Write out stats per split
        # stats_per_split = df.groupby('split')[numeric_columns].agg(['mean', 'std'])
        # stats_per_split.reset_index()
        # self.out_stats_per_split.write(data=stats_per_split, index=True)
