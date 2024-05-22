# from typing import Dict, List, NamedTuple, Callable
# from pathlib import Path
# from abc import ABC, abstractmethod
# import logging

# from functools import partial
# from multiprocessing import Pool, Manager

# from omegaconf import DictConfig
# from tqdm import tqdm

# from .executor_base import BaseStageExecutor
# from . import GenericHandler

# # from omegaconf import DictConfig

# # from ...core import (
# #     BaseStageExecutor,
# #     ExperimentParameters,
# #     Asset,
# #     GenericHandler
# #     )
# # from ..px_statistics import get_func
# # # from src.utils.make_a_map import make_a_map

# logger = logging.getLogger(__name__)


# class FrozenAsset(NamedTuple):
#     path: Path
#     handler: GenericHandler


# class TaskTarget(NamedTuple):
#     asset: FrozenAsset
#     norm_factors: float
#     split_name: str
#     sim_num: str
#     freq: int


# class ParallelExecutor(BaseStageExecutor):
#     def __init__(self, cfg: DictConfig, stage_str: str) -> None:
#         super().__init__(cfg, stage_str)

#         # self.out_report: Asset = self.assets_out["report"]
#         # self.in_cmb_map_true: Asset = self.assets_in["cmb_map"]
#         # self.in_cmb_map_pred: Asset = self.assets_in["cmb_map_post"]

#         # self.epochs = list(cfg.training.analysis.epoch)

#         # self.stat_func_dict = self.cfg.analysis.px_functions
#         # self.stat_funcs = self.get_stat_funcs()

#         self.num_processes = cfg.local_system.num_processes

#     def execute(self) -> None:
#         logger.debug("ParallelExecutor execute() method.")

#         process = partial(process_target, stat_funcs=self.stat_funcs)
#         # Tasks are items on a to-do list
#         #   For each simulation, we compare the prediction and target
#         #   A task contains labels, file names, and handlers for each sim
#         tasks = self.build_tasks()

#         # Run a single task outside multiprocessing to catch issues quickly.
#         self.try_a_task(process, tasks[0])

#         self.run_all_tasks(process, tasks)

#     def run_all_tasks(self, process, tasks):
#         # Use multiprocessing to search through sims in parallel
#         # A manager allows collection of information across separate threads
#         with Manager() as manager:
#             results = manager.list()
#             # The Pool sets up the individual processes. 
#             # Set processes according to the capacity of your computer
#             with Pool(processes=self.num_processes) as pool:
#                 # Each result is the output of "process" running on each of the tasks
#                 for result in tqdm(pool.imap_unordered(process, tasks), total=len(tasks)):
#                     results.append(result)
#             # Convert the results to a regular list after multiprocessing is complete
#             #     and before the scope of the manager ends
#             results_list = list(results)
#         # Use the out_report asset to write all results to disk
#         self.out_report.write(results_list)

#     def build_tasks(self):
#         tasks = []
#         for split in self.splits:
#             with self.name_tracker.set_context("split", split.name):
#                 for sim in split.iter_sims():
#                     with self.name_tracker.set_context("sim_num", sim):
#                         for epoch in self.epochs:
#                             with self.name_tracker.set_context("epoch", epoch):
#                                 true = self.in_cmb_map_true
#                                 assert true.path.exists(), f"Cannot find: {true.path}"  # Better to fail now than in the pool
#                                 true = FrozenAsset(path=true.path, handler=true.handler)
                                
#                                 pred = self.in_cmb_map_pred
#                                 assert pred.path.exists(), f"Cannot find: {pred.path}"  # Better to fail now than in the pool
#                                 pred = FrozenAsset(path=pred.path, handler=pred.handler)
                                
#                                 tasks.append(TaskTarget(true_asset=true,
#                                                         pred_asset=pred,
#                                                         split_name=split.name, 
#                                                         sim_num=sim,
#                                                         epoch=epoch))
#         return tasks

#     def try_a_task(self, process, task: TaskTarget):
#         """
#         Get statistics for one sim (task) outside multiprocessing first, 
#         to avoid painful debugging within multiprocessing.
#         """
#         # def save_map(self, map_asset: FrozenAsset, fn):
#         #     handler = map_asset.handler
#         #     map_data = handler.read(map_asset.path)[0]
#         #     make_a_map(map_data, fn, title=fn)
#         # self.save_map(task.true_asset, "true.png")
#         # self.save_map(task.pred_asset, "pred.png")
#         res = process(task)
#         if 'error' in res.keys():
#             raise Exception(res['error'])


# def process_target(task_target: TaskTarget, stat_funcs):
#     """
#     Each stat_func should accept true, pred, and **kwargs to catch other things
#     """
#     true = task_target.true_asset
#     true_data = true.handler.read(true.path)
#     pred = task_target.pred_asset
#     pred_data = pred.handler.read(pred.path)

#     res = {'split': task_target.split_name, 'sim': task_target.sim_num, 'epoch':task_target.epoch}
#     try:
#         for stat_name, func in stat_funcs.items():
#             res[stat_name] = func(true_data, pred_data)
#     except Exception as e:
#         # TODO: Ensure 'this X is defined in Y' stays accurate. Or relabel it.
#         res['error'] = f"Running '{stat_name}' caused {str(e)}. This stat function is defined in px_statistics.yaml."

#     return res
