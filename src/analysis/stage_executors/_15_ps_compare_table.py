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
        self.baseline = self.get_baseline_for_comparison()
        self.labels_lookup = self.get_labels_lookup()

    def get_labels_lookup(self):
        lookup = dict(self.cfg.model.analysis.ps_functions)
        return lookup

    def get_baseline_for_comparison(self):
        cfg_val = self.cfg.ps_baseline
        if cfg_val == "theory":
            return "thry"
        elif cfg_val == "theory":
            return "real"
        return cfg_val

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute().")
        all_summaries = []

        multi_epoch = True
        # Get each summary table
        for model_comp in self.models_to_compare:
            summary = self.gather_from(model_comp)
            epochs = model_comp.get("epochs", None)
            if epochs is None:
                epoch = model_comp.get("epoch")
                epoch = int(-1) if epoch == "" else epoch
                epochs = [epoch]
                multi_epoch = False
            filtered_summary = summary[summary["epoch"].isin(epochs)]
            filtered_summary = summary[summary["baseline"] == self.baseline]
            all_summaries.append(filtered_summary)

        # Combine summaries into a single table
        summaries = pd.concat(all_summaries, ignore_index=True)
        summaries = summaries.drop(columns = "baseline")
        if not multi_epoch:
            summaries = summaries.drop(columns = "epoch")

        summary_table = self.create_summary_table(summaries, multi_epoch)
        relabel_map = {k: v['label'] for k, v in self.labels_lookup.items()}
        summary_table.rename(columns=relabel_map, inplace=True)
        summary_table = summary_table.sort_index(axis=1)

        console_table = self.format_summary_table(summary_table, latex=False)
        logger.info(f"Power Spectrum Comparison: \n\n{console_table}\n")

        latex_table = self.format_summary_table(summary_table, latex=True)
        num_index_levels = latex_table.index.nlevels
        num_data_columns = len(latex_table.columns)
        column_format = "l" * num_index_levels + "c" * num_data_columns
        latex_table = latex_table.to_latex(escape=False,
                                           caption="Power Spectrum Performance",
                                           label="tab:px_metrics", 
                                           column_format=column_format)
        latex_table = latex_table.replace("\\begin{table}",
                                          "\\begin{table}\n\\centering")
        logger.info(f"Power Spectrum Comparison: \n\n{latex_table}\n")
        self.out_report.write(data=latex_table)

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

    def create_summary_table(self, summary, multi_epoch=False):
        if multi_epoch:
            index = ['model_name', 'epoch', 'baseline']
        else:
            index = ['model_name']
        df_pivot = summary.pivot_table(index=index, 
                                       columns=['metric', 'type'], 
                                       values='value', 
                                       aggfunc='first')
        return df_pivot

    def format_summary_table(self, summary_table, latex):
        formatted_df = pd.DataFrame(index=summary_table.index)

        metrics = [col[0] for col in summary_table.columns]

        # Iterate over each metric in the MultiIndex DataFrame
        for metric in metrics:
            mean_col = (metric, 'mean')
            std_col = (metric, 'std')

            # Apply the custom formatting function to each row
            formatted_df[metric] = summary_table.apply(
                lambda row: format_mean_std(row[mean_col],
                                            row[std_col],
                                            sig_digs=4,
                                            latex=latex),
                                            axis=1)
        return formatted_df
