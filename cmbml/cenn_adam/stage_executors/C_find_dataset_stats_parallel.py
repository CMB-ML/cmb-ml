from typing import Dict, List, NamedTuple, Callable
import logging
from pathlib import Path

from functools import partial
from multiprocessing import Pool, Manager

import numpy as np

from omegaconf import DictConfig
from tqdm import tqdm

from cmbml.core import (
    BaseStageExecutor,
    Asset,
    GenericHandler
    )
from cmbml.utils import make_instrument, Instrument
from cmbml.core.asset_handlers import Config # Import for typing hint
from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap # Import for typing hint


logger = logging.getLogger(__name__)


class FrozenAsset(NamedTuple):
    path: Path
    handler: GenericHandler


class TaskTarget(NamedTuple):
    cmb_asset: FrozenAsset
    obs_asset: FrozenAsset
    split_name: str
    sim_num: str


class FindDatasetStatsParallelExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="get_dataset_stats")

        self.instrument: Instrument = make_instrument(cfg=cfg)
        self.channels = self.instrument.dets.keys()

        self.out_dataset_stats: Asset = self.assets_out["dataset_stats"]
        out_norm_handler: Config

        self.in_cmb_map: Asset = self.assets_in["cmb_map"]
        self.in_obs_maps: Asset = self.assets_in["obs_maps"]
        in_cmb_map_handler: HealpyMap
        in_obs_map_handler: HealpyMap

        scaling = cfg.model.patch_nn.get("scaling", None)
        if scaling and scaling != "minmax":
            msg = f"Only minmax scaling is supported, not {scaling}."
            raise NotImplementedError(msg)
        self.get_extrema = scaling == "minmax"

        self.scale_scan_method = None
        self.scale_sift_method = None
        self.set_scale_find_methods()

        self.n_workers = cfg.model.patch_nn.preprocess.n_workers

    def set_scale_find_methods(self):
        # A different method can be used to find alternative statistics:
        scan_method = find_min_max  # Replace as needed
        self.scale_scan_method = partial(scan_method,
                                         freqs=self.channels, 
                                         map_fields=self.map_fields)
        # A different function can be used to find, e.g. the abs(min, max):
        sift_method = sift_min_max_results  # Replace as needed
        self.scale_sift_method = partial(sift_method,
                                         instrument=self.instrument, 
                                         map_fields=self.map_fields)

    def execute(self) -> None:
        if not self.get_extrema:
            logger.warning("Model yaml does not request extrema file. Skipping.")
            return

        logger.debug(f"Running {self.__class__.__name__} execute().")
        # Tasks are items on a to-do list
        #   For each simulation, we compare the prediction and target
        #   A task contains labels, file names, and handlers for each sim
        tasks = self.build_tasks()

        # Run a single task outside multiprocessing to catch issues quickly.
        self.try_a_task(self.scale_scan_method, tasks[0])

        results_list = self.run_all_tasks(self.scale_scan_method, tasks)

        results_summary = self.scale_sift_method(results_list)

        self.out_dataset_stats.write(data=results_summary)


    def run_all_tasks(self, process, tasks):
        # Use multiprocessing to search through sims in parallel
        # A manager allows collection of information across separate threads
        with Manager() as manager:
            results = manager.list()
            # The Pool sets up the individual processes. 
            # Set processes according to the capacity of your computer
            with Pool(processes=self.n_workers) as pool:
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


# These functions can be used in multiprocessing.
# They can also be swapped out for different functions, e.g. to find the abs(min, max) or the average instead.
# Side note: you can't find the average and standard deviation for all simulations in the same function,
#    because the average and standard deviation are not additive (yay statistics!).
def find_min_max(task_target: TaskTarget, freqs, map_fields):
    """
    Acts on a single simulation (TaskTarget) to find the max and min values
        for each detector and field.
    """
    cmb = task_target.cmb_asset
    cmb_data = cmb.handler.read(cmb.path)
    obs = task_target.obs_asset

    res = {'cmb': {}}
    for i, map_field in enumerate(map_fields):
        res['cmb'][map_field] = {'vmin': cmb_data[i,:].min(), 
                                 'vmax': cmb_data[i,:].max()}

    for freq in freqs:
        obs_data = obs.handler.read(str(obs.path).format(freq=freq))
        res[freq] = {}
        # In case a simulation has 3 map fields, but the current settings are for just 1
        #    Use zip to stop early if there are fewer map_fields 
        #    (this may be a bit confusing, consider alternatives)
        for i, _ in zip(range(obs_data.shape[0]), map_fields):
            map_field = map_fields[i]
            res[freq][map_field] = {'vmin': obs_data[i,:].min(), 'vmax': obs_data[i,:].max()}
    return res


def sift_min_max_results(results_list, instrument: Instrument, map_fields:str):
    """
    Sifts through aggregated results to find the min and max over all simulations
    Uses sift_for_detector
    """
    summary = {'cmb': {}}
    for det in instrument.dets.values():
        freq = det.nom_freq
        summary[freq] = {}
        for map_field in det.fields:
            summary[freq][map_field] = {}
            vmin, vmax = sift_for_detector(results_list, freq=freq, map_field=map_field)
            summary[freq][map_field]['vmin'] = vmin
            summary[freq][map_field]['vmax'] = vmax
    for map_field in map_fields:
        summary['cmb'][map_field] = {}
        vmin, vmax = sift_for_detector(results_list, freq='cmb', map_field=map_field)
        summary['cmb'][map_field]['vmin'] = vmin
        summary['cmb'][map_field]['vmax'] = vmax
    return summary


def sift_for_detector(results_list, freq, map_field):
    """
    Gets the min and max for a single detector
    """
    min_vals = [d[freq][map_field]['vmin'] for d in results_list]
    max_vals = [d[freq][map_field]['vmax'] for d in results_list]
    return min(min_vals), max(max_vals)