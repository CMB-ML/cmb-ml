from typing import Dict, List, NamedTuple, Callable
import logging
from pathlib import Path

from functools import partial
from multiprocessing import Pool, Manager

from tqdm import tqdm

from omegaconf import DictConfig

from cmbml.core import BaseStageExecutor, Asset, AssetWithPathAlts, GenericHandler
from cmbml.analysis.px_statistics import get_func
from cmbml.core.asset_handlers.pd_csv_handler import PandasCsvHandler # Import for typing hint
from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap # Import for typing hint
from cmbml.core.asset_handlers.psmaker_handler import NumpyPowerSpectrum

logger = logging.getLogger(__name__)


class FrozenAsset(NamedTuple):
    path: Path
    handler: GenericHandler


class TaskTarget(NamedTuple):
    pred_asset: FrozenAsset
    base_asset: FrozenAsset
    baseline_label: str
    split_name: str
    sim_num: str
    epoch: int


class PSAnalysisExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="ps_analysis")

        self.out_report: Asset = self.assets_out["report"]
        out_report_handler: PandasCsvHandler

        self.in_ps_theory: AssetWithPathAlts = self.assets_in["theory_ps"]
        self.in_ps_real: Asset = self.assets_in["auto_real"]
        self.in_ps_pred: Asset = self.assets_in["auto_pred"]
        in_ps_handler: NumpyPowerSpectrum

        self.stat_func_dict = self.cfg.model.analysis.ps_functions
        self.stat_funcs = self.get_stat_funcs()

        self.num_processes = cfg.model.analysis.ps_operations.num_processes

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute().")
        
        # Create a method; 
        #   process_target needs a list of statistics functions, from the config file
        #   partial() makes a `process` that takes a single argument
        process = partial(process_target, stat_funcs=self.stat_funcs)
        # Tasks are items on a to-do list
        #   For each simulation, we compare the prediction and target
        #   A task contains labels, file names, and handlers for each sim
        tasks = self.build_tasks()

        # Run a single task outside multiprocessing to catch issues quickly.
        self.try_a_task(process, tasks[0])

        self.run_all_tasks(process, tasks)

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
        self.review_report(results_list)
        # Use the out_report asset to write all results to disk
        self.out_report.write(data=results_list)

    def review_report(self, report_list):
        found_error = False
        for res in report_list:
            if 'error' in res.keys():
                logger.error(f"Error in split {res['split']}, sim {res['sim']}, epoch {res['epoch']}: {res['error']}")
                found_error = True
        if found_error:
            raise OSError("Errors were found in the report. Please review the log for details.")

    def build_tasks(self):
        tasks = []
        for split in self.splits:
            for sim in split.iter_sims():
                for epoch in self.model_epochs:
                    context = dict(split=split.name, sim_num=sim, epoch=epoch)
                    with self.name_tracker.set_contexts(contexts_dict=context):
                        pred = self.in_ps_pred
                        pred = FrozenAsset(path=pred.path, handler=pred.handler)

                        real = self.in_ps_real
                        real = FrozenAsset(path=real.path, handler=real.handler)

                        thry = self.in_ps_theory
                        thry = FrozenAsset(path=thry.path, handler=thry.handler)

                        tasks.append(TaskTarget(pred_asset=pred,
                                                base_asset=thry,
                                                baseline_label="thry",
                                                split_name=split.name, 
                                                sim_num=sim,
                                                epoch=epoch))

                        tasks.append(TaskTarget(pred_asset=pred,
                                                base_asset=real,
                                                baseline_label="real",
                                                split_name=split.name, 
                                                sim_num=sim,
                                                epoch=epoch))

        return tasks

    def try_a_task(self, process, task: TaskTarget):
        """
        Get statistics for one sim (task) outside multiprocessing first, 
        to avoid painful debugging within multiprocessing.
        """
        res = process(task)
        if 'error' in res.keys():
            raise Exception(res['error'])

    def get_stat_funcs(self):
        stat_funcs = {}
        for name, details in self.stat_func_dict.items():
            func = get_func(details["func"])
            if 'kwargs' in details:
                func = partial(func, **details['kwargs'])
            stat_funcs[name] = func
        return stat_funcs


def process_target(task_target: TaskTarget, stat_funcs):
    """
    Each stat_func should accept true, pred, and **kwargs to catch other things
    """
    res = {'split': task_target.split_name, 'sim': task_target.sim_num, 'epoch':task_target.epoch}
    res = dict(
        split=task_target.split_name,
        sim=task_target.sim_num,
        epoch=task_target.epoch,
        baseline=task_target.baseline_label
    )
    pred = task_target.pred_asset
    base = task_target.base_asset
    try:
        true_data = base.handler.read(base.path)
    except OSError as e:
        return {'error': f"Could not read true data from {base.path}. Error: {str(e)}", **res}
    try:
        pred_data = pred.handler.read(pred.path)
    except OSError as e:
        return {'error': f"Could not read pred data from {pred.path}. Error: {str(e)}", **res}

    # Ensure that the shapes match
    if pred_data.shape[0] < true_data.shape[0]:
        # If so, use just the portion of true data needed.
        true_data = true_data[:pred_data.shape[0]]

    try:
        for stat_name, func in stat_funcs.items():
            res[stat_name] = func(true_data, pred_data)
    except Exception as e:
        res['error'] = f"Running '{stat_name}' caused '{str(e)}'. This stat function is defined in stat_funcs.yaml."

    return res
