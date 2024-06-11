import logging

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from omegaconf import DictConfig

from cmbml.core import BaseStageExecutor, Asset
from cmbml.core.asset_handlers.pd_csv_handler import PandasCsvHandler # Import for typing hint
from cmbml.core.asset_handlers.txt_handler import TextHandler # Import for typing hint
from cmbml.utils.number_report import format_mean_std


logger = logging.getLogger(__name__)


class PixelCompareFigExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="pixel_comparison")

        self.in_report: Asset = self.assets_in["overall_stats"]
        in_report_handler: PandasCsvHandler
        
        self.out_report: Asset = self.assets_out["latex_report"]
        out_report_handler: TextHandler

        self.models_to_compare = cfg.models_comp
        self.labels_lookup = self.get_labels_lookup()

    def get_labels_lookup(self):
        lookup = dict(self.cfg.model.analysis.px_functions)
        return lookup

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute().")
        all_summaries = []
        for model_comp in self.models_to_compare:
            epoch = model_comp["epoch"]
            working_directory = model_comp["working_directory"]
            all_summaries.append(self.gather_from(working_directory, epoch))

        # Create for the console & log
        st = self.create_summary_table(all_summaries)
        relabel_map = {k: v['label'] for k, v in self.labels_lookup.items()}
        st.rename(columns=relabel_map, inplace=True)
        st = st.sort_index(axis=1)
        logger.info(f"Pixel Comparison: \n\n{st}\n")

        # Create for LaTeX
        st = self.create_summary_table(all_summaries, latex=True)
        st.rename(columns=relabel_map, inplace=True)
        st = st.sort_index(axis=1)
        column_format = "l" + "c" * (len(st.columns))
        latex_table = st.to_latex(escape=False, 
                                  caption="Pixel Space Performance", 
                                  label="tab:px_metrics", 
                                  column_format=column_format)
        latex_table = latex_table.replace("\\begin{table}", "\\begin{table}\n\\centering")
        logger.info(f"Pixel Comparison (LaTeX): \n\n{latex_table}")
        self.out_report.write(data=latex_table)

    def gather_from(self, working_directory, epoch):
        context_params = dict(working=working_directory, epoch=epoch)
        with self.name_tracker.set_contexts(context_params):
            summary = self.in_report.read()
        return summary

    def create_summary_table(self, all_summaries, latex=False):
        results = {}
        
        for model_comp, summary in zip(self.models_to_compare, all_summaries):
            model_name = model_comp["model_name"]
            mean_values = summary.iloc[0, 1:]    # Skip junk column 1
            std_values = summary.iloc[1, 1:]

            # Format the results as 'mean +/- std' or the latex equivalent
            results[model_name] = [format_mean_std(mean, std, latex=latex) 
                                   for mean, std in zip(mean_values, std_values)]
        
        # Create a DataFrame from the results dictionary
        result_df = pd.DataFrame(results).transpose()
        result_df.columns = summary.columns[1:]
        
        return result_df
