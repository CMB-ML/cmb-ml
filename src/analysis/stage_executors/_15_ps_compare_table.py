import logging
from functools import partial

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from omegaconf import DictConfig, OmegaConf

from src.core import BaseStageExecutor, Asset
from src.core.asset_handlers.pd_csv_handler import PandasCsvHandler # Import for typing hint
from src.core.asset_handlers.txt_handler import TextHandler # Import for typing hint
from src.utils.number_report import format_mean_std

logger = logging.getLogger(__name__)


class PSCompareTableExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="ps_comparison")

        self.in_report: Asset = self.assets_in["epoch_stats"]
        in_report_handler: PandasCsvHandler
        
        self.out_report: Asset = self.assets_out["latex_report"]
        out_report_handler: TextHandler

        self.models_to_compare = cfg.models_comp
        self.labels_lookup = self.get_labels_lookup()

    def get_labels_lookup(self):
        lookup = dict(self.cfg.model.analysis.ps_functions)
        return lookup

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute().")
        all_summaries = []

        # Get each summary table
        for model_comp in self.models_to_compare:
            summary = self.gather_from(model_comp)
            all_summaries.append(summary)

        # Combine summaries into a single table
        summaries = pd.concat(all_summaries, ignore_index=True)

        summary_table = self.create_summary_table(summaries)
        relabel_map = {k: v['label'] for k, v in self.labels_lookup.items()}
        summary_table.rename(columns=relabel_map, inplace=True)
        summary_table = summary_table.sort_index(axis=1)

        self.print_console_table(summary_table, relabel_map.values())
        self.print_latex_table(summary_table, relabel_map.values())

    def gather_from(self, model_comp):
        model_dict = OmegaConf.to_container(model_comp, resolve=True)
        working_directory = model_dict["working_directory"]

        context_params = dict(working=working_directory)
        with self.name_tracker.set_contexts(context_params):
            summary = self.in_report.read()
        summary['model_name'] = model_dict['model_name']
        summary['epoch'] = summary['epoch'].fillna(-1).astype(int)
        # If we want to get particular epochs, filter the summary table
        epochs = model_dict.get('epochs', None)
        if epochs is None:
            epoch = model_dict.get('epoch', None)
            epoch = int(-1) if epoch == "" else epoch
            epochs = [epoch]
        if epochs is not None:
            summary = summary[summary['epoch'].isin(epochs)]
        return summary

    def create_summary_table(self, summary):
        df_pivot = summary.pivot_table(index=['model_name', 'epoch', 'baseline'], 
                                       columns=['metric', 'type'], 
                                       values='value', 
                                       aggfunc='first')
        return df_pivot

    def print_console_table(self, summary_table, metrics):
        # Print the table to the console
        _apply_formatting = partial(apply_formatting, metrics=metrics, latex=False)
        df_formatted = summary_table.apply(_apply_formatting, axis=1)
        df_formatted = df_formatted.reset_index()

        # relabel_map = {k: v['label'] for k, v in self.labels_lookup.items()}
        # summary_table.rename(columns=relabel_map, inplace=True)
        # summary_table = summary_table.sort_index(axis=1)
        logger.info(f"Power Spectrum Comparison: \n\n{summary_table}\n")

    def print_latex_table(self, summary_table, metrics):
        _apply_formatting = partial(apply_formatting, metrics=metrics, latex=True)
        df_formatted = summary_table.apply(_apply_formatting, axis=1)
        df_formatted = df_formatted.reset_index()

        # Print the table to LaTeX
        # relabel_map = {k: v['label'] for k, v in self.labels_lookup.items()}
        # summary_table.rename(columns=relabel_map, inplace=True)
        # summary_table = summary_table.sort_index(axis=1)
        # column_format = "l" + "c" * (len(summary_table.columns))
        column_format = "l" + "c" * (len(summary_table.columns))
        latex_table = summary_table.to_latex(escape=False, 
                                             caption="Power Spectrum Performance", 
                                             label="tab:ps_metrics", 
                                             column_format=column_format)
        latex_table = latex_table.replace("\\begin{table}", "\\begin{table}\n\\centering")
        logger.info(f"Power Spectrum Comparison (LaTeX): \n\n{latex_table}")
        self.out_report.write(data=latex_table)

def apply_formatting(row, metrics, latex):
    # Apply the formatting to each row
    formatted_metrics = {}
    for metric in metrics:
        mean = row[(metric, 'mean')]
        std = row[(metric, 'std')]
        formatted_metrics[metric] = format_mean_std(mean, std, latex=latex)
    return pd.Series(formatted_metrics)
