# from typing import List, Dict, NamedTuple
# from pathlib import Path
# import logging

# import numpy as np

# from multiprocessing import Pool

# from omegaconf import DictConfig
# from tqdm import tqdm

# from core import (
#     BaseStageExecutor, 
#     Split,
#     GenericHandler,
#     Asset
#     )
# from core.asset_handlers.asset_handlers_base import Config
# from core.asset_handlers.healpy_map_handler import HealpyMap
# from src.utils import make_instrument, Instrument, Detector


# logger = logging.getLogger(__name__)


# class FrozenAsset(NamedTuple):
#     path: Path
#     handler: GenericHandler


# class TaskTarget(NamedTuple):
#     asset_in: FrozenAsset
#     asset_out: FrozenAsset
#     all_map_fields: str
#     detector_fields: str
#     norm_factors: float


# class PreprocessExecutor(BaseStageExecutor):
#     def __init__(self, cfg: DictConfig) -> None:
#         # The following string must match the pipeline yaml
#         super().__init__(cfg, stage_str="preprocess")

#         self.out_cmb_asset: Asset = self.assets_out["cmb_map"]
#         self.out_obs_assets: Asset = self.assets_out["obs_maps"]
#         out_cmb_map_handler: HealpyMap
#         out_obs_map_handler: HealpyMap

#         self.in_norm_file: Asset = self.assets_in["norm_file"]
#         self.in_cmb_asset: Asset = self.assets_in["cmb_map"]
#         self.in_obs_assets: Asset = self.assets_in["obs_maps"]
#         in_norm_file_handler: Config
#         in_cmb_map_handler: HealpyMap
#         in_obs_map_handler: HealpyMap

#         self.instrument: Instrument = make_instrument(cfg=cfg)
#         self.num_processes = self.cfg.model.petroff.preprocess.num_processes

#     def execute(self) -> None:
#         logger.debug(f"Running {self.__class__.__name__} execute().")
#         # Tasks are items on a to-do list
#         #   For each simulation, we compare the prediction and target
#         #   A task contains labels, file names, and handlers for each sim
#         tasks = self.build_tasks()

#         # Run the first task outside multiprocessing for easier debugging.
#         first_task = tasks.pop(0)
#         self.try_a_task(parallel_preprocess, first_task)

#         self.run_all_tasks(parallel_preprocess, tasks)

#     def build_tasks(self):
#         scale_factors = self.in_norm_file.read()
#         tasks = []
#         for split in self.splits:
#             for sim in split.iter_sims():
#                 context = dict(split=split.name, sim_num=sim)
#                 with self.name_tracker.set_contexts(contexts_dict=context):
#                     cmb_asset_in = self.make_frozen_asset(self.in_cmb_asset)
#                     cmb_asset_out = self.make_frozen_asset(self.out_cmb_asset)
#                     x = TaskTarget(
#                         asset_in=cmb_asset_in,
#                         asset_out=cmb_asset_out,
#                         all_map_fields=self.cfg.scenario.map_fields,
#                         detector_fields=self.cfg.scenario.map_fields,
#                         norm_factors=scale_factors['cmb'],
#                     )
#                 tasks.append(x)
#                 for freq, detector in self.instrument.dets.items():
#                     context['freq'] = freq
#                     with self.name_tracker.set_contexts(contexts_dict=context):
#                         f_asset_in = self.make_frozen_asset(self.in_obs_assets)
#                         f_asset_out = self.make_frozen_asset(self.out_obs_assets)
#                         x = TaskTarget(
#                             asset_in=f_asset_in,
#                             asset_out=f_asset_out,
#                             all_map_fields=self.cfg.scenario.map_fields,
#                             detector_fields=detector.fields,
#                             norm_factors=scale_factors[freq],
#                         )
#                     tasks.append(x)
#         return tasks

#     @staticmethod
#     def make_frozen_asset(asset: Asset):
#         return FrozenAsset(path=asset.path, handler=asset.handler)

#     def try_a_task(self, _process, task: TaskTarget):
#         """
#         Clean one map outside multiprocessing,
#         to avoid painful debugging within multiprocessing.
#         """
#         _process(task)
#         logger.info(f"First simulation preprocessed by {self.__class__.__name__} without errors.")

#     def run_all_tasks(self, process, tasks):
#         logger.info(f"Running preprocess on {len(tasks)} tasks across {self.num_processes} workers.")
#         with Pool(processes=self.num_processes) as pool:
#             # Create an iterator from imap_unordered and wrap it with tqdm for progress tracking
#             task_iterator = tqdm(pool.imap_unordered(process, tasks), total=len(tasks))
#             # Iterate through the task_iterator to execute the tasks
#             for _ in task_iterator:
#                 pass


# def parallel_preprocess(task_target: TaskTarget):
#     tt = task_target
#     in_asset = tt.asset_in
#     in_map = in_asset.handler.read(in_asset.path)

#     prepped_map = preprocess_map(
#         all_map_fields=tt.all_map_fields,
#         map_data=in_map,
#         scale_factors=tt.norm_factors,
#         detector_fields=tt.detector_fields
#     )

#     out_asset = tt.asset_out
#     out_asset.handler.write(path=out_asset.path, data=prepped_map)


# def preprocess_map(all_map_fields: str, 
#                    map_data: np.ndarray, 
#                    scale_factors: Dict[str, Dict[str, float]],
#                    detector_fields: str
#                    ) -> List[np.ndarray]:
#     processed_maps = []
#     all_fields:str = all_map_fields  # Either I or IQU
#     for field_char in detector_fields:
#         field_idx = all_fields.find(field_char)
#         field_scale = scale_factors[field_char]
#         field_data = map_data[field_idx]
#         scaled_map = normalize(field_data, field_scale)
#         processed_maps.append(scaled_map)
#     return processed_maps


# def normalize(in_map, extrema):
#     norm_min = extrema['min_val']
#     norm_max = extrema['max_val']
#     out_map = (in_map - norm_min) / (norm_max - norm_min)
#     return out_map
