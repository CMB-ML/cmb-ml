# from typing import Union
# import logging

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from scipy.stats import pearsonr
# import json

# from omegaconf import DictConfig


# from cmbml.core import (
#     BaseStageExecutor, 
#     Split,
#     Asset, AssetWithPathAlts
#     )
# from tqdm import tqdm

# from cmbml.core.asset_handlers.asset_handlers_base import EmptyHandler, Config # Import for typing hint
# from cmbml.core.asset_handlers.psmaker_handler import NumpyPowerSpectrum


# logger = logging.getLogger(__name__)


# class PowerSpectrumAnalysisExecutorSerial(BaseStageExecutor):
#     def __init__(self, cfg: DictConfig) -> None:
#         # The following string must match the pipeline yaml
#         super().__init__(cfg, stage_str="ps_analysis2")

#         self.out_wmap_distribution: Asset = self.assets_out["wmap_distribution"]
#         out_distribution_handler: EmptyHandler

#         self.out_error_distribution: Asset = self.assets_out["error_distribution"]
#         out_error_distribution_handler: EmptyHandler

#         self.out_ps_report: Asset = self.assets_out["report"]
#         out_ps_report_handler: EmptyHandler

#         self.in_ps_theory: AssetWithPathAlts = self.assets_in["theory_ps"]
#         self.in_ps_real: Asset = self.assets_in["auto_real"]
#         self.in_ps_pred: Asset = self.assets_in["auto_pred"]
#         in_ps_handler: NumpyPowerSpectrum

#         self.n_ps = self.get_stage_element("wmap_n_ps")
#         self.fig_n_override = self.override_sim_nums

#     def execute(self) -> None:
#         # Remove this function
#         logger.debug(f"Running {self.__class__.__name__} execute()")
#         mean_ps, std_ps, first_band, second_band = self.generate_wmap_distribution(self.n_ps)
#         for split in self.splits:
#             self.process_split(split, mean_ps, std_ps, first_band, second_band)
    
#     def generate_wmap_distribution(self, n_ps):
#         ps_idx = np.arange(0, n_ps)

#         # number of ells
#         ell_max = self.get_test_shape()

#         ps_samples = np.zeros((n_ps, ell_max))

#         for i, idx in enumerate(ps_idx):
#             with self.name_tracker.set_contexts({"split": "Train", "sim_num": idx}):
#                 ps_theory = self.in_ps_theory.read(use_alt_path=False)
#             ps_samples[i] = ps_theory[:ell_max]
        
#         mean_ps = np.mean(ps_samples, axis=0)
#         std_ps = np.std(ps_samples, axis=0)

#         first_band = ((mean_ps - std_ps), (mean_ps + std_ps))
#         second_band = ((mean_ps - 2 * std_ps), (mean_ps + 2 * std_ps))

#         return mean_ps, std_ps, first_band, second_band
    
#     def generate_error_bands(self, errors):
#         mean = np.mean(errors, axis=0)
#         std = np.std(errors, axis=0)

#         first_band = ((mean - std), (mean + std))
#         second_band = ((mean - 2 * std), (mean + 2 * std))

#         return mean, std, first_band, second_band

#     def get_test_shape(self):
#         # Get shape of real PS from the first test sim 
#         # TODO: Make sure to rework when multiple test splits exist
#         with self.name_tracker.set_contexts({"split": "Test", "sim_num": 0}):
#             ps_real = self.in_ps_real.read()
#         # Return the number of ell values
#         return ps_real.shape[0]
    
#     def process_split(self, split: Split, wmap_mean, wmap_std, first_band, second_band) -> None:
#         logger.info(f"Running {self.__class__.__name__} process_split() for split: {split.name}.")
#         if split.name == "Train":
#             self.process_train(wmap_mean, wmap_std)
#         else:
#             with self.name_tracker.set_context("split", split.name):
#                 self.process_test(split, first_band, second_band)

#     def process_train(self, wmap_mean, wmap_std) -> None:
#         res = {}

#         res["wmap_mean"] = wmap_mean.tolist()
#         res["wmap_std"] = wmap_std.tolist()

#         wmap_path = self.out_wmap_distribution.path
#         self.out_wmap_distribution.write()
#         with open(wmap_path, "w") as f:
#             json.dump(res, f)
        
#         logger.debug(f"Saved WMAP distribution to {wmap_path}")

#     def process_test(self, split:Split, first_band, second_band) -> None:
#         logger.info(f"Running {self.__class__.__name__} process_test() for split: {split.name}.")

#         # We may want to process a subset of all sims
#         if self.fig_n_override is None:
#             sim_iter = split.iter_sims()
#         else:
#             sim_iter = self.fig_n_override

#         if split.ps_fidu_fixed:
#             ps_theory = self.in_ps_theory.read(use_alt_path=True)
#         else:
#             ps_theory = None

#         split_res = []
#         errors = []

#         for sim in tqdm(sim_iter):
#             with self.name_tracker.set_context("sim_num", sim):
#                 error, res = self.process_sim(ps_theory, first_band, second_band)
#                 split_res = split_res + res
#                 errors = errors + error
#         self.out_ps_report.write()
#         pd.DataFrame(split_res).to_csv(self.out_ps_report.path, index=False)

#         error_data = self.process_df(pd.DataFrame(errors))

#         error_path = self.out_error_distribution.path
#         with open(error_path, "w") as f:
#             json.dump(error_data, f)

#     def process_df(self, df: pd.DataFrame):
#         res = {}
#         col_vals = df['epoch'].unique()
#         for val in col_vals:
#             epoch_df = df[df['epoch'] == val]
#             # concat all errors for each epoch
#             errors = np.vstack(epoch_df['error'].values)
#             mean, std, _, _ = self.generate_error_bands(errors)
#             res[f'{val}_mean'] = mean.tolist()
#             res[f'{val}_std'] = std.tolist()
#         return res

#     def process_sim(self, ps_theory, first_band, second_band) -> None:
#         sim_res = []
#         errors = []
#         for epoch in self.model_epochs:
#             with self.name_tracker.set_context("epoch", epoch):
#                 error, res = self.process_epoch(ps_theory, first_band, second_band)
#                 sim_res.append(res)
#                 errors.append(error)
#         return errors, sim_res

#     def process_epoch(self, ps_theory, first_band, second_band) -> None:
#         epoch = self.name_tracker.context['epoch']
#         split = self.name_tracker.context['split']
#         sim_num = self.name_tracker.context['sim_num']
#         ps_real = self.in_ps_real.read()
#         ps_pred = self.in_ps_pred.read()

#         ells = np.arange(1, ps_real.shape[0] + 1)

#         norm = ells * (ells + 1) / (2 * np.pi)

#         ps_real = ps_real * norm
#         ps_pred = ps_pred * norm

#         error = ps_pred - ps_real
#         error_dict = {'epoch': epoch, 'split': split, 'sim_num': sim_num, 'error': error}
#         if ps_theory is None:
#             ps_theory = self.in_ps_theory.read(use_alt_path=False)[:ps_real.shape[0]]

#         res = {'split': split, 'sim_num': sim_num, 'epoch': epoch}

#         real_bounds = self.within_both_bounds(ps_real, first_band, second_band, 'real')
#         res.update(real_bounds)

#         pred_bounds = self.within_both_bounds(ps_pred, first_band, second_band, 'pred')
#         res.update(pred_bounds)
        
#         pred_real_metrics = self.calculate_metrics(ps_pred, ps_real, 'pred_real')
#         res.update(pred_real_metrics)

#         pred_theory_metrics = self.calculate_metrics(ps_pred, ps_theory, 'pred_theory')
#         res.update(pred_theory_metrics)

#         return (error_dict, res)
    
#     def calculate_metrics(self, observed, target, prefix) -> dict:
#         mae = np.mean(np.abs(observed - target))
#         mse = np.mean((observed - target) ** 2)
#         rmse = np.sqrt(mse)
#         corr, _ = pearsonr(observed, target)
#         return {f'{prefix}_mae': mae, f'{prefix}_mse': mse, f'{prefix}_rmse': rmse, f'{prefix}_corr': corr}
    
#     def within_both_bounds(self, arr, first_band, second_band, prefix):
#         first_count = self.half_within_range(arr, first_band)
#         second_count = self.half_within_range(arr, second_band)
#         return {f'{prefix}_half_one_sigma': first_count, f'{prefix}_half_two_sigma': second_count}

#     def half_within_range(self, arr, band):
#         lower_bound, upper_bound = band
#         within_range = (arr >= lower_bound) & (arr <= upper_bound)
#         majority_count = np.sum(within_range) > (0.5 * len(arr))
#         return majority_count