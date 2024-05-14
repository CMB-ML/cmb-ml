from typing import NamedTuple

from pathlib import Path
from typing import List, Dict
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from multiprocessing import Pool

from omegaconf import DictConfig

from core import (
    BaseStageExecutor, 
    Split,
    GenericHandler,
    AssetWithPathAlts
    )
# from ..make_ps import get_power as _get_power
from core.asset_handlers.psmaker_handler import CambPowerSpectrum, NumpyPowerSpectrum


logger = logging.getLogger(__name__)


class FrozenAsset(NamedTuple):
    # FrozenAsset is created as an immutable so that multiprocessing can run.
    path: Path
    handler: GenericHandler


class TaskTarget(NamedTuple):
    asset_in: FrozenAsset
    asset_out: FrozenAsset


class ConvertTheoryPowerSpectrumExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="convert_theory_ps")

        self.out_theory_ps: AssetWithPathAlts = self.assets_out["theory_ps"]
        out_theory_ps_handler: NumpyPowerSpectrum

        self.in_theory_ps: AssetWithPathAlts = self.assets_in["theory_ps"]
        in_theory_ps_handler: CambPowerSpectrum

        self.num_processes = cfg.model.analysis.px_operations.num_processes

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute().")

        # Tasks are items on a to-do list
        #   For each simulation, we compare the prediction and target
        #   A task contains labels, file names, and handlers for each sim
        tasks = self.build_tasks()

        # Run the first task outside multiprocessing for easier debugging.
        first_task = tasks.pop(0)
        self.try_a_task(parallel_convert, first_task)

        self.run_all_tasks(parallel_convert, tasks)

    def build_tasks(self):
        tasks = []
        for split in self.splits:
            with self.name_tracker.set_context("split", split):
                if split.ps_fidu_fixed:
                    task = self.build_a_task(split.ps_fidu_fixed)
                    tasks.append(task)
                    continue
                for sim in split.iter_sims():
                    with self.name_tracker.set_context("sim_num", sim):
                        task = self.build_a_task(split.ps_fidu_fixed)
                        tasks.append(task)
        return tasks

    def build_a_task(self, use_path_alt):
        ps_in = self.make_frozen_asset(self.in_theory_ps, use_path_alt)
        ps_out = self.make_frozen_asset(self.out_theory_ps, use_path_alt)
        task = TaskTarget(asset_in=ps_in, asset_out=ps_out)
        return task

    @staticmethod
    def make_frozen_asset(asset: AssetWithPathAlts, use_path_alt: bool):
        path = asset.path_alt if use_path_alt else asset.path
        frozen_asset = FrozenAsset(path=path, handler=asset.handler)
        return frozen_asset

    def try_a_task(self, _process, task: TaskTarget):
        """
        Clean one map outside multiprocessing,
        to avoid painful debugging within multiprocessing.
        """
        _process(task)

    def run_all_tasks(self, process, tasks):
        with Pool(processes=self.num_processes) as pool:
            # Create an iterator from imap_unordered and wrap it with tqdm for progress tracking
            task_iterator = tqdm(pool.imap_unordered(process, tasks), total=len(tasks))
            # Iterate through the task_iterator to execute the tasks
            for _ in task_iterator:
                pass


def parallel_convert(task_target: TaskTarget):
    tt = task_target
    in_asset = tt.asset_in
    in_ps = in_asset.handler.read(in_asset.path)

    out_asset = tt.asset_out
    out_ps = out_asset.handler.write(path=out_asset.path, data=in_ps)
