import logging

import numpy as np
from omegaconf import DictConfig

from cmbml.core import Split, Asset, BaseStageExecutor
from cmbml.core.asset_handlers import NpyHandler
from cmbml.utils.parse_ell_list import parse_ells
import cmbml.utils.split_metrics as spm


logger = logging.getLogger(__name__)


class GetPerSimMetricsExecutor(BaseStageExecutor):
    """Per-realization scalars: one value per simulation, shape (N,).

    The orthogonal reduction to GetPerEllMetricsExecutor. That one collapses 
    over realizations to get (n_ell,); this collapses over ell to get (N,),
    so a figure can rank realizations and a table can quote a spread across 
    the split.

    Built on split_metrics' per-instance intermediates rather than a fresh
    expression, so a number here cannot disagree with the per-ell artifact.
    """
    def __init__(self, cfg: DictConfig, stage_str="get_per_sim_metrics") -> None:
        super().__init__(cfg, stage_str=stage_str)
        self.out_metric: Asset = self.assets_out["per_sim_metric"]
        out_metric_handler: NpyHandler
        self.in_pred: Asset = self.assets_in["pred"]
        self.in_thry: Asset = self.assets_in["thry"]

        self.ells = parse_ells(cfg.model.analysis.use_ells)
        self.include_metrics = self.get_stage_element("metrics")

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        for split in self.splits:
            for epoch in self.model_epochs:
                with self.name_tracker.set_contexts(
                        dict(split=split.name, epoch=epoch)):
                    self.process(split)
            logger.info(f"Done getting per-sim metrics for {split.name}.")

    def process(self, split: Split) -> None:
        pred = np.atleast_2d(self.in_pred.read())
        thry = np.atleast_2d(self.in_thry.read())

        metrics = self.compute(pred, thry)

        unknown = [m for m in self.include_metrics if m not in metrics]
        if unknown:
            raise ValueError(
                f"{self.stage_str} asks for {unknown}; "
                f"available: {sorted(metrics)}"
            )

        for name in self.include_metrics:
            with self.name_tracker.set_context("metric", name):
                self.out_metric.write(data=metrics[name])

    def compute(self, pred, thry) -> dict:
        """Everything reducible to one value per realization."""
        se = spm.squared_error(pred, thry)          # (N, n_ell)
        re = spm.rel_err(pred, thry)                # (N, n_ell)
        return {
            "mse": se.mean(1),
            "rmse": np.sqrt(se.mean(1)),
            "abs_rel_err_pct": np.abs(re).mean(1) * 100,
            "bias": (pred - thry).mean(1),
        }
