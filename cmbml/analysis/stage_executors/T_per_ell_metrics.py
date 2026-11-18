import logging

import numpy as np
from omegaconf import DictConfig

from cmbml.core import Split, Asset, BaseStageExecutor
from cmbml.core.asset_handlers import NpyHandler
from cmbml.utils.parse_ell_list import parse_ells
from cmbml.utils.split_metrics import compute_metrics

logger = logging.getLogger(__name__)


class GetPerEllMetricsExecutor(BaseStageExecutor):
    """
    Calculate per-multipole metrics from bundled power spectra.
    """
    def __init__(self, cfg: DictConfig, stage_str="get_per_ell_metrics") -> None:
        super().__init__(cfg, stage_str=stage_str)
        self.out_metric: Asset = self.assets_out["per_ell_metric"]
        out_metric_handler: NpyHandler
        self.in_pred: Asset = self.assets_in["pred"]
        self.in_thry: Asset = self.assets_in["thry"]
        self.in_mean_thry: Asset = self.assets_in["mean_thry"]
        in_pred_handler: NpyHandler
        in_thry_handler: NpyHandler
        in_mean_thry_handler: NpyHandler

        self.ells = parse_ells(cfg.model.analysis.use_ells)

        self.include_metrics = self.get_stage_element("metrics")

        self.sky_fraction = None
        if self.sky_fraction is not None:
            # Don't set sky_fraction to something! This is broken! Return to this!
            raise NotImplementedError("The default: 1, assumes no mask. We're using a mask. There's big questions here to address.")

        for banned_metric in ["cosmic_var", "treble_cosmic_var", "e_cv"]:
            if banned_metric in self.include_metrics:
                # There's an open question about how to use the mask and get the proper fsky for mode counting.
                raise NotImplementedError("These metrics depend on sky fraction. fsky needs to be handled.")

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        for split in self.splits:
            for epoch in self.model_epochs:
                context = dict(split=split.name, epoch=epoch)
                with self.name_tracker.set_contexts(context):
                    self.process(split)
            logger.info(f"Done getting metrics for {split.name}.")

    def process(self, split: Split) -> None:
        pred = self.in_pred.read()
        thry = self.in_thry.read()
        mean_thry = self.in_mean_thry.read()

        fsky = 1 if self.sky_fraction is None else self.sky_fraction
        metrics = compute_metrics(pred, thry, ells=self.ells, fsky=fsky, mean_theory=mean_thry)

        for m in self.include_metrics:
            with self.name_tracker.set_context("metric", m):
                self.out_metric.write(data=metrics[m])
