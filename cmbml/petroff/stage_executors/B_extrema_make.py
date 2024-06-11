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
from cmbml.utils import make_instrument, Instrument
from cmbml.core.asset_handlers.asset_handlers_base import Config # Import for typing hint
from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap # Import for typing hint
from cmbml.petroff.preprocessing.scale_tasks_helper import TaskTarget, FrozenAsset
from cmbml.petroff.preprocessing.scale_methods_factory import get_sim_scanner, get_sim_sifter

logger = logging.getLogger(__name__)



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

        self.scale_scan_method = None
        self.scale_sift_method = None
        self.set_scale_find_methods()

        self.num_processes = cfg.model.petroff.preprocess.num_processes

    def set_scale_find_methods(self):
        scale_method_name = self.cfg.model.petroff.preprocess.scaling
        scan_method = get_sim_scanner(method=scale_method_name)
        self.scale_scan_method = partial(scan_method,
                                         freqs=self.channels, 
                                         map_fields=self.map_fields)
        sift_method = get_sim_sifter(method=scale_method_name)
        self.scale_sift_method = partial(sift_method,
                                         instrument=self.instrument, 
                                         map_fields=self.map_fields)

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute().")
        # Tasks are items on a to-do list
        #   For each simulation, we compare the prediction and target
        #   A task contains labels, file names, and handlers for each sim
        tasks = self.build_tasks()

        # Run a single task outside multiprocessing to catch issues quickly.
        self.try_a_task(self.scale_scan_method, tasks[0])

        results_list = self.run_all_tasks(self.scale_scan_method, tasks)

        results_summary = self.scale_sift_method(results_list)

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
