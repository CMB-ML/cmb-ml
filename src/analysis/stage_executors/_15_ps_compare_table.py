import logging

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from omegaconf import DictConfig

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
        for model_comp in self.models_to_compare:
            epoch = model_comp["epoch"]
            working_directory = model_comp["working_directory"]
            summary = self.gather_from(working_directory, epoch)
            all_summaries.append((model_comp['model_name'], summary))

        for summary in all_summaries:
            print(summary)

        # # Create for LaTeX output
        # st = self.create_summary_table(all_summaries, latex=True)
        # st.rename(columns=relabel_map, inplace=True)
        # st = st.sort_index(axis=1)
        # column_format = "l" + "c" * (len(st.columns))
        # latex_table = st.to_latex(escape=False, 
        #                           caption="Power Spectrum Performance", 
        #                           label="tab:ps_metrics", 
        #                           column_format=column_format)
        # latex_table = latex_table.replace("\\begin{table}", "\\begin{table}\n\\centering")
        # logger.info(f"Power Spectrum Comparison (LaTeX): \n\n{latex_table}")
        # self.out_report.write(data=latex_table)

    def gather_from(self, working_directory, epoch):
        context_params = dict(working=working_directory, epoch=epoch)
        with self.name_tracker.set_contexts(context_params):
            summary = self.in_report.read()
        # Filter data for the specific epoch and other relevant filters
        summary = summary[(summary['epoch'] == epoch)]
        return summary

    # def gather_from(self, working_directory, epoch):
    #     context_params = dict(working=working_directory, epoch=epoch)
    #     with self.name_tracker.set_contexts(context_params):
    #         summary = self.in_report.read()
    #     return summary

    def create_summary_table(self, all_summaries, latex=False):
        results = {}
        
        for model_name, summary in all_summaries:
            data = {}
            for idx, row in summary.iterrows():
                key = (row['metric'], row['type'])  # Combine metric and type ('mean' or 'std')
                data[key] = row['value']
                
            # Format the results as 'mean +/- std' or the latex equivalent
            formatted_results = {metric: format_mean_std(data[(metric, 'mean')], 
                                                         data[(metric, 'std')], 
                                                         latex=latex)
                                 for metric in set(m for m, _ in data.keys())}
            results[model_name] = formatted_results
        
        # Create a DataFrame from the results dictionary
        result_df = pd.DataFrame(results).transpose()
        
        return result_df

    # def create_summary_table(self, all_summaries, latex=False):
    #     results = {}
        
    #     for model_comp, summary in zip(self.models_to_compare, all_summaries):
    #         model_name = model_comp["model_name"]
    #         mean_values = summary.iloc[0, 1:]    # Skip junk columns in column 1 and the last column
    #         std_values = summary.iloc[1, 1:]

    #         # Format the results as 'mean +/- std' or the latex equivalent
    #         results[model_name] = [format_mean_std(mean, std, latex=latex) 
    #                                for mean, std in zip(mean_values, std_values)]
        
    #     # Create a DataFrame from the results dictionary
    #     result_df = pd.DataFrame(results).transpose()
    #     result_df.columns = summary.columns[1:]
        
    #     return result_df
