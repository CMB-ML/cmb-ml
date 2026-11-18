"""
Scalar evaluation metrics aggregated over specified multipole ranges.
"""
import logging

import numpy as np
from omegaconf import DictConfig

from cmbml.core import Asset, BaseStageExecutor, Split
from cmbml.core.asset_handlers import NpyHandler, Config
from cmbml.utils.parse_ell_list import parse_ells


logger = logging.getLogger(__name__)


class GetScalarMetricsExecutor(BaseStageExecutor):
    """
    Calculate scalar evaluation metrics from saved per-multipole metrics.

    All metrics are calculated over both the full and common multipole
    ranges, regardless of method. Table membership is handled separately.

    Aggregation follows the manuscript's Evaluation Metrics definitions.
    In particular, skill is calculated from aggregated MSE values rather
    than by averaging per-multipole skill.

    Undefined metrics are represented by NaN.
    """

    INPUT_METRICS = (
        "mse",
        "abs_rel_err_pct",
        "R2",
        "baseline_mse",
        "bias",
        "resid_var",
        "tension",
        "r",
        "A",
    )

    def __init__(
        self,
        cfg: DictConfig,
        stage_str="get_scalar_metrics",
    ) -> None:
        super().__init__(cfg, stage_str=stage_str)

        self.out_scalar_metrics: Asset = self.assets_out["scalar_metrics"]
        out_metric_handler: Config

        self.in_metric: Asset = self.assets_in["per_ell_metric"]
        in_metric_handler: NpyHandler

        self.ells = np.asarray(parse_ells(cfg.model.analysis.use_ells))
        self.lcut = int(self.get_stage_element("lcut"))

        if self.ells.ndim != 1:
            raise ValueError(
                f"Multipole axis must be 1D, got {self.ells.shape}."
            )

        if not np.all(np.diff(self.ells) == 1):
            raise ValueError(
                "Scalar evaluation requires a consecutive multipole axis."
            )

        if not self.ells[0] <= self.lcut <= self.ells[-1]:
            raise ValueError(
                f"lcut={self.lcut} outside multipole range "
                f"[{self.ells[0]}, {self.ells[-1]}]."
            )

        self.ranges = {
            "full": np.ones(len(self.ells), dtype=bool),
            "common": self.ells <= self.lcut,
        }

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

        for split in self.splits:
            for epoch in self.model_epochs:
                with self.name_tracker.set_contexts(
                    dict(split=split.name, epoch=epoch)
                ):
                    self.process(split)

            logger.info(
                f"Done getting scalar metrics for {split.name}."
            )

    def process(self, split: Split) -> None:
        """
        Load all required per-multipole metrics and aggregate
        over both comparison ranges.

        Results are saved together in one YAML configuration file.
        """
        per_ell = {}

        for metric in self.INPUT_METRICS:
            with self.name_tracker.set_context("metric", metric):
                values = np.asarray(self.in_metric.read())

            if values.shape != self.ells.shape:
                raise ValueError(
                    f"Metric {metric}: expected shape "
                    f"{self.ells.shape}, got {values.shape}."
                )

            per_ell[metric] = values

        results = {
            "ranges": {
                "full": [int(self.ells[0]), int(self.ells[-1])],
                "common": [int(self.ells[0]), int(self.lcut)],
            },
            "metrics": {},
        }

        for range_name, mask in self.ranges.items():
            results["metrics"][range_name] = self.compute(per_ell, mask)

        self.out_scalar_metrics.write(data=results)

    @staticmethod
    def compute(per_ell: dict, mask: np.ndarray) -> dict:
        """
        Calculate scalar summaries for one multipole range.

        Uses the arithmetic mean over multipoles, except where the
        manuscript explicitly defines a different aggregation.

        Undefined per-multipole quantities propagate to the scalar
        result rather than being silently omitted.
        """
        def avg(metric):
            return np.mean(per_ell[metric][mask])

        mse = avg("mse")
        baseline_mse = avg("baseline_mse")

        if baseline_mse > 0:
            skill = 1.0 - mse / baseline_mse
        else:
            skill = np.nan

        bias = per_ell["bias"][mask]
        resid_var = per_ell["resid_var"][mask]
        amplitude = per_ell["A"][mask]

        B = np.sqrt(np.mean(bias**2))
        V = np.sqrt(np.mean(resid_var))

        return {
            # Predictive skill and accuracy
            "mse": mse,
            "abs_rel_err_pct": avg("abs_rel_err_pct"),
            "R2": avg("R2"),
            "skill": skill,

            # Bias and scatter
            "B": B,
            "V": V,
            "tension": avg("tension"),

            # Sensitivity
            "r": avg("r"),
            "delta_A": np.mean(np.abs(amplitude - 1.0)),
        }
