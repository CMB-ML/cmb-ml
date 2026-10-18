"""Pixel-space CMB reconstruction statistics across target beam resolutions."""

import logging
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import NamedTuple

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from cmbml.analysis.statistics import get_func
from cmbml.core import Asset, BaseStageExecutor, GenericHandler
from cmbml.core.asset_handlers import Config
from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap  # Register handler
from cmbml.utils.beam_naming import name_beam_fwhms

logger = logging.getLogger(__name__)


class FrozenAsset(NamedTuple):
    path: Path
    handler: GenericHandler


class TaskTarget(NamedTuple):
    true_asset: FrozenAsset
    pred_asset: FrozenAsset
    split_name: str
    sim_num: int
    epoch: int | str
    beam_name: str


# Initialized once per worker rather than passing the (large) mask per task.
_worker_mask = None
_worker_stat_funcs = None


def _init_worker(mask, stat_funcs):
    global _worker_mask, _worker_stat_funcs
    _worker_mask = mask
    _worker_stat_funcs = stat_funcs


def _process_worker(task):
    return process_target(task, stat_funcs=_worker_stat_funcs, mask=_worker_mask)


class PixelAnalysisExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="pixel_analysis")

        self.out_report: Asset = self.assets_out["report"]
        out_report_handler: Config

        self.in_cmb_map_true: Asset = self.assets_in["cmb_map_sim"]
        self.in_cmb_map_pred: Asset = self.assets_in["cmb_map_post"]
        self.in_mask: Asset = self.assets_in["mask"]
        in_cmb_map_handler: HealpyMap
        in_mask_handler: HealpyMap

        self.stat_func_dict = cfg.model.analysis.px_functions
        self.stat_funcs = self.get_stat_funcs()
        self.num_processes = cfg.model.analysis.px_operations.num_processes
        self.target_beams = name_beam_fwhms(cfg.model.analysis.target_beam_fwhm)

    def execute(self) -> None:
        # Sharp analysis mask: True means include this pixel.
        mask_data = self.in_mask.read(map_fields=self.in_mask.use_fields)
        mask = np.asarray(mask_data[0] if np.ndim(mask_data) == 2 else mask_data, dtype=bool)
        if mask.ndim != 1 or not np.any(mask):
            raise ValueError("Sharp analysis mask must be a nonempty 1D HEALPix mask")

        tasks = self.build_tasks()
        if not tasks:
            logger.warning("No pixel-analysis tasks were generated")
            self.out_report.write(data=[])
            return

        # Debug one task in the main process before launching workers.
        self.try_a_task(tasks[0], mask)
        self.run_all_tasks(tasks, mask)

    def build_tasks(self):
        tasks = []
        for split in self.splits:
            for sim in split.iter_sims():
                for epoch in self.model_epochs:
                    for beam_name, _ in self.target_beams:
                        context = dict(split=split.name, sim_num=sim,
                                       epoch=epoch, beam_name=beam_name)
                        with self.name_tracker.set_contexts(contexts_dict=context):
                            true = self.in_cmb_map_true
                            pred = self.in_cmb_map_pred
                            tasks.append(TaskTarget(
                                true_asset=FrozenAsset(true.path, true.handler),
                                pred_asset=FrozenAsset(pred.path, pred.handler),
                                split_name=split.name,
                                sim_num=sim,
                                epoch=epoch,
                                beam_name=beam_name,
                            ))
        return tasks

    def try_a_task(self, task: TaskTarget, mask: np.ndarray) -> None:
        result = process_target(task, stat_funcs=self.stat_funcs, mask=mask)
        if "error" in result:
            raise RuntimeError(result["error"])

    def run_all_tasks(self, tasks, mask: np.ndarray) -> None:
        # imap_unordered already returns results to the parent; no Manager needed.
        with Pool(processes=self.num_processes,
                  initializer=_init_worker, initargs=(mask, self.stat_funcs)) as pool:
            results = list(tqdm(pool.imap_unordered(_process_worker, tasks),
                                total=len(tasks)))
        self.review_report(results)
        self.out_report.write(data=results)

    def review_report(self, results) -> None:
        failures = [res for res in results if "error" in res]
        for res in failures:
            logger.error("Error in split %s, sim %s, epoch %s, beam %s: %s",
                         res["split"], res["sim"], res["epoch"],
                         res["beam_name"], res["error"])
        if failures:
            raise OSError(f"{len(failures)} pixel-analysis tasks failed; see logs")

    def get_stat_funcs(self):
        stat_funcs = {}
        for name, details in self.stat_func_dict.items():
            func = get_func(details["func"])
            if "kwargs" in details:
                func = partial(func, **details["kwargs"])
            stat_funcs[name] = func
        return stat_funcs


def _numerical_array(data):
    """Keep numerical map values; support Astropy/PySM quantities."""
    return np.asarray(data.value if hasattr(data, "value") else data)


def process_target(task_target: TaskTarget, stat_funcs, mask: np.ndarray):
    """Compute configured statistics on the sharp-mask-selected pixels."""
    result = dict(split=task_target.split_name, sim=task_target.sim_num,
                  epoch=task_target.epoch, beam_name=task_target.beam_name)

    try:
        true_data = _numerical_array(
            task_target.true_asset.handler.read(task_target.true_asset.path))
        pred_data = _numerical_array(
            task_target.pred_asset.handler.read(task_target.pred_asset.path))
    except Exception as exc:
        return {**result, "error": f"Could not read maps: {exc}"}

    # Normalize scalar temperature maps to (field, pixel) arrays.
    if true_data.ndim == 1:
        true_data = true_data[np.newaxis, :]
    if pred_data.ndim == 1:
        pred_data = pred_data[np.newaxis, :]

    if true_data.ndim != 2 or pred_data.ndim != 2:
        return {**result, "error": f"Unsupported map shapes: {true_data.shape}, {pred_data.shape}"}

    if true_data.shape[1] != pred_data.shape[1] or true_data.shape[1] != mask.size:
        return {**result, "error":
                f"Pixel-count mismatch: true={true_data.shape}, "
                f"pred={pred_data.shape}, mask={mask.shape}"}

    # Preserve legacy behavior: allow prediction of fewer Stokes fields
    # than the ground truth, but not more.
    if pred_data.shape[0] > true_data.shape[0]:
        return {**result, "error":
                f"Prediction has more fields than ground truth: "
                f"true={true_data.shape}, pred={pred_data.shape}"}
    true_data = true_data[:pred_data.shape[0]]

    # Select the same sky pixels for all fields, then flatten as before.
    true_values = true_data[:, mask].reshape(-1)
    pred_values = pred_data[:, mask].reshape(-1)

    for stat_name, func in stat_funcs.items():
        try:
            result[stat_name] = float(func(true_values, pred_values))
        except Exception as exc:
            result["error"] = (f"Running '{stat_name}' caused '{exc}'. "
                               "This function is defined in stat_funcs.yaml.")
            break
    return result
