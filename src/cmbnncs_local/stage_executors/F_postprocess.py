from typing import Dict, List, NamedTuple, Callable
from pathlib import Path
from abc import ABC, abstractmethod
import logging

import numpy as np

from multiprocessing import Pool, Manager

from omegaconf import DictConfig
from tqdm import tqdm

from core import (
    BaseStageExecutor,
    GenericHandler,
    Split,
    Asset
    )
from cmbnncs.spherical import piecePlanes2spheres
from ..handler_npymap import NumpyMap             # Import to register the AssetHandler
from core.asset_handlers.asset_handlers_base import Config # Import for typing hint
from core.asset_handlers.healpy_map_handler import HealpyMap # Import for typing hint
from src.utils import make_instrument, Instrument, Detector


logger = logging.getLogger(__name__)


class FrozenAsset(NamedTuple):
    path: Path
    handler: GenericHandler


class TaskTarget(NamedTuple):
    asset_in: FrozenAsset
    asset_out: FrozenAsset
    all_map_fields: str
    norm_factors: float


class PostprocessExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # TODO: remove self.stage_str; pass it as a parameter to the super.init()
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="postprocess")

        self.instrument: Instrument = make_instrument(cfg=cfg)

        self.out_cmb_asset: Asset = self.assets_out["cmb_map"]
        out_cmb_map_handler: HealpyMap

        self.in_norm_file: Asset = self.assets_in["norm_file"]
        self.in_cmb_asset: Asset = self.assets_in["cmb_map"]
        in_norm_file_handler: Config
        in_cmb_map_handler: NumpyMap

        self.num_processes = self.cfg.model.cmbnncs.postprocess.num_processes

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute().")

        # Tasks are items on a to-do list
        #   For each simulation, we compare the prediction and target
        #   A task contains labels, file names, and handlers for each sim
        tasks = self.build_tasks()

        # Run the first task outside multiprocessing for easier debugging.
        first_task = tasks.pop(0)
        self.try_a_task(parallel_postprocess, first_task)

        self.run_all_tasks(parallel_postprocess, tasks)

    def build_tasks(self):
        scale_factors = self.in_norm_file.read()
        tasks = []
        for epoch in self.model_epochs:
            for split in self.splits:
                for sim in split.iter_sims():
                    context = dict(split=split.name, sim_num=sim, epoch=epoch)
                    with self.name_tracker.set_contexts(contexts_dict=context):
                        cmb_asset_in = self.make_frozen_asset(self.in_cmb_asset)
                        cmb_asset_out = self.make_frozen_asset(self.out_cmb_asset)
                        x = TaskTarget(
                            asset_in=cmb_asset_in,
                            asset_out=cmb_asset_out,
                            all_map_fields=self.cfg.scenario.map_fields,
                            norm_factors=scale_factors['cmb'],
                        )
                    tasks.append(x)
        return tasks

    @staticmethod
    def make_frozen_asset(asset: Asset):
        return FrozenAsset(path=asset.path, handler=asset.handler)

    def try_a_task(self, _process, task: TaskTarget):
        """
        Clean one map outside multiprocessing,
        to avoid painful debugging within multiprocessing.
        """
        _process(task)
        logger.info(f"First simulation postprocessed by {self.__class__.__name__} without errors.")

    def run_all_tasks(self, process, tasks):
        logger.info(f"Running postprocess on {len(tasks)} tasks across {self.num_processes} workers.")
        with Pool(processes=self.num_processes) as pool:
            # Create an iterator from imap_unordered and wrap it with tqdm for progress tracking
            task_iterator = tqdm(pool.imap_unordered(process, tasks), total=len(tasks))
            # Iterate through the task_iterator to execute the tasks
            for _ in task_iterator:
                pass


def parallel_postprocess(task_target: TaskTarget):
    tt = task_target
    in_asset = tt.asset_in
    in_map = in_asset.handler.read(in_asset.path)

    prepped_map = postprocess_map(
        all_map_fields=tt.all_map_fields,
        map_data=in_map,
        scale_factors=tt.norm_factors,
    )

    out_asset = tt.asset_out
    out_asset.handler.write(path=out_asset.path, data=prepped_map, column_units=["uK_CMB"])


def postprocess_map(all_map_fields: str, 
                map_data: np.ndarray, 
                scale_factors: Dict[str, Dict[str, float]],
                ) -> List[np.ndarray]:
    processed_maps = []
    for field_char in all_map_fields:
        field_idx = all_map_fields.find(field_char)
        field_scale = scale_factors[field_char]
        field_data = map_data[field_idx]
        scaled_map = unnormalize(field_data, field_scale)
        mangled_map = piecePlanes2spheres(scaled_map)
        processed_maps.append(mangled_map)
    return processed_maps


def unnormalize(in_map, scale_factors):
    scale = scale_factors['scale']
    out_map = in_map * scale
    return out_map
