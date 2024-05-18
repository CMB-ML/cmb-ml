from typing import Dict, List, NamedTuple, Callable
import logging
from pathlib import Path

from functools import partial
from multiprocessing import Pool, Manager

import numpy as np

from omegaconf import DictConfig
from tqdm import tqdm

from core import (
    BaseStageExecutor,
    Asset,
    GenericHandler
    )
from src.utils import make_instrument, Instrument
from core.asset_handlers.asset_handlers_base import Config # Import for typing hint
from core.asset_handlers.healpy_map_handler import HealpyMap # Import for typing hint

logger = logging.getLogger(__name__)


class FrozenAsset(NamedTuple):
    path: Path
    handler: GenericHandler


class TaskTarget(NamedTuple):
    cmb_asset: FrozenAsset
    obs_asset: FrozenAsset
    split_name: str
    sim_num: str


class PreprocessMakeExtremaExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="make_normalization")

        self.instrument: Instrument = make_instrument(cfg=cfg)
        self.channels = self.instrument.dets.keys()

        self.out_norm_file: Asset = self.assets_out["norm_file"]
        out_norm_handler: Config

        self.in_cmb_map: Asset = self.assets_in["cmb_map"]
        self.in_obs_maps: Asset = self.assets_in["obs_maps"]
        in_cmb_map_handler: HealpyMap
        in_obs_map_handler: HealpyMap

        self.num_processes = cfg.model.petroff.prep.num_processes

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute().")
        
        # Create a method; 
        #   process_target needs a list of statistics functions, from the config file
        #   partial() makes a `process` that takes a single argument
        process = partial(find_abs_max, 
                          freqs=self.channels, 
                          map_fields=self.map_fields)
        # Tasks are items on a to-do list
        #   For each simulation, we compare the prediction and target
        #   A task contains labels, file names, and handlers for each sim
        tasks = self.build_tasks()

        # Run a single task outside multiprocessing to catch issues quickly.
        self.try_a_task(process, tasks[0])

        results_list = self.run_all_tasks(process, tasks)

        results_summary = self.sift_abs_max_results(results_list)

        self.out_norm_file.write(data=results_summary)


    def run_all_tasks(self, process, tasks):
        # Use multiprocessing to search through sims in parallel
        # A manager allows collection of information across separate threads
        with Manager() as manager:
            results = manager.list()
            # The Pool sets up the individual processes. 
            # Set processes according to the capacity of your computer
            with Pool(processes=self.num_processes) as pool:
                # Each result is the output of "process" running on each of the tasks
                for result in tqdm(pool.imap_unordered(process, tasks), total=len(tasks)):
                    results.append(result)
            # Convert the results to a regular list after multiprocessing is complete
            #     and before the scope of the manager ends
            results_list = list(results)
        # Use the out_report asset to write all results to disk
        return results_list


    def build_tasks(self):
        tasks = []
        for split in self.splits:
            for sim in split.iter_sims():
                context = dict(split=split.name, sim_num=sim)
                with self.name_tracker.set_contexts(contexts_dict=context):
                    cmb = self.in_cmb_map
                    cmb = FrozenAsset(path=cmb.path, handler=cmb.handler)
                    
                    obs = self.in_obs_maps
                    with self.name_tracker.set_context("freq", "{freq}"):
                        obs = FrozenAsset(path=obs.path, handler=obs.handler)
                    
                    tasks.append(TaskTarget(cmb_asset=cmb,
                                            obs_asset=obs,
                                            split_name=split.name, 
                                            sim_num=sim))
        return tasks

    def try_a_task(self, process, task: TaskTarget):
        """
        Get statistics for one sim (task) outside multiprocessing first, 
        to avoid painful debugging within multiprocessing.
        """
        res = process(task)
        if 'error' in res.keys():
            raise Exception(res['error'])

    # def sift_min_max_results(self, results_list):
    #     summary = {'cmb': {}}
    #     for det in self.instrument.dets.values():
    #         freq = det.nom_freq
    #         summary[freq] = {}
    #         for map_field in det.fields:
    #             summary[freq][map_field] = {}
    #             min_v, max_v = self.sift_min_max_result(results_list, freq=freq, map_field=map_field)
    #             summary[freq][map_field]['min_val'] = min_v
    #             summary[freq][map_field]['max_val'] = max_v
    #     for map_field in self.map_fields:
    #         summary['cmb'][map_field] = {}
    #         min_v, max_v = self.sift_min_max_result(results_list, freq='cmb', map_field=map_field)
    #         summary['cmb'][map_field]['min_val'] = min_v
    #         summary['cmb'][map_field]['max_val'] = max_v
    #     return summary

    # def sift_min_max_result(self, results_list, freq, map_field):
    #     min_vals = [d[freq][map_field]['min_v'] for d in results_list]
    #     max_vals = [d[freq][map_field]['max_v'] for d in results_list]
    #     return min(min_vals), max(max_vals)

    def sift_abs_max_results(self, results_list):
        summary = {'cmb': {}}
        for det in self.instrument.dets.values():
            freq = det.nom_freq
            summary[freq] = {}
            for map_field in det.fields:
                summary[freq][map_field] = {}
                max_v = self.sift_abs_max_result(results_list, freq=freq, map_field=map_field)
                summary[freq][map_field]['abs_max'] = max_v
        for map_field in self.map_fields:
            summary['cmb'][map_field] = {}
            max_v = self.sift_abs_max_result(results_list, freq='cmb', map_field=map_field)
            summary['cmb'][map_field]['abs_max'] = max_v
        return summary

    def sift_abs_max_result(self, results_list, freq, map_field):
        max_vals = [d[freq][map_field]['abs_max'] for d in results_list]
        return max(max_vals)



def find_abs_max(task_target: TaskTarget, freqs, map_fields):
    """
    Each stat_func should accept true, pred, and **kwargs to catch other things
    """
    cmb = task_target.cmb_asset
    cmb_data = cmb.handler.read(cmb.path)
    obs = task_target.obs_asset

    res = {'cmb': {}}
    for i, map_field in enumerate(map_fields):
        abs_max = np.max(np.abs(cmb_data[i,:]))
        res['cmb'][map_field] = {'abs_max': abs_max}

    for freq in freqs:
        obs_data = obs.handler.read(str(obs.path).format(freq=freq))
        res[freq] = {}
        # In case a simulation hsa 3 map fields, but the current settings are for just 1
        for i, _ in zip(range(obs_data.shape[0]), map_fields):
            map_field = map_fields[i]
            abs_max = np.max(np.abs(obs_data[i,:]))
            res[freq][map_field] = {'abs_max': abs_max}
    return res


def find_min_max(task_target: TaskTarget, freqs, map_fields):
    """
    Each stat_func should accept true, pred, and **kwargs to catch other things
    """
    cmb = task_target.cmb_asset
    cmb_data = cmb.handler.read(cmb.path)
    obs = task_target.obs_asset

    res = {'cmb': {}}
    for i, map_field in enumerate(map_fields):
        res['cmb'][map_field] = {'min_v': cmb_data[i,:].min(), 
                                 'max_v': cmb_data[i,:].max()}

    for freq in freqs:
        obs_data = obs.handler.read(str(obs.path).format(freq=freq))
        res[freq] = {}
        # In case a simulation hsa 3 map fields, but the current settings are for just 1
        for i, _ in zip(range(obs_data.shape[0]), map_fields):
            map_field = map_fields[i]
            res[freq][map_field] = {'min_v': obs_data[i,:].min(), 'max_v': obs_data[i,:].max()}
    return res



