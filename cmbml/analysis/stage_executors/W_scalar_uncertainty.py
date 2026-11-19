"""Uncertainty summaries for the multipole-range ps scalar metrics.

Standard deviations describe across-simulation variability of individual MSE
and MARE. Bootstrap percentile intervals describe sampling uncertainty of
all defined scalar estimators. Bootstrap samples use matched prediction-target
pairs; a fixed seed ensures identical resampling indices across methods for
the same split size and ordering.
"""
import logging

import numpy as np
from omegaconf import DictConfig

from cmbml.core import Asset, BaseStageExecutor, Split
from cmbml.core.asset_handlers import Config, NpyHandler
from cmbml.utils.parse_ell_list import parse_ells
from cmbml.utils.split_metrics import compute_metrics
from cmbml.analysis.stage_executors.U_scalar_metrics import GetScalarMetricsExecutor


logger = logging.getLogger(__name__)


class GetScalarUncertaintyExecutor(BaseStageExecutor):
    """Write across-simulation SDs and bootstrap CIs for both multipole ranges."""

    def __init__(self, cfg: DictConfig,
                 stage_str="get_scalar_uncertainty") -> None:
        super().__init__(cfg, stage_str=stage_str)
        self.out_uncertainty: Asset = self.assets_out["uncertainty"]
        out_metric_handler: Config

        self.in_pred: Asset = self.assets_in["pred"]
        self.in_thry: Asset = self.assets_in["thry"]
        self.in_mean_thry: Asset = self.assets_in["mean_thry"]
        in_pred_handler: NpyHandler
        in_thry_handler: NpyHandler
        in_mean_thry_handler: NpyHandler

        self.ells = np.asarray(parse_ells(cfg.model.analysis.use_ells))
        self.lcut = int(self.get_stage_element("lcut"))
        self.n_bootstrap = int(self.get_stage_element("n_bootstrap"))
        self.seed = int(self.get_stage_element("seed"))
        self.confidence = float(self.get_stage_element("confidence"))
        if self.ells.ndim != 1 or self.ells.size == 0:
            raise ValueError("Expected a nonempty one-dimensional ell axis.")
        if not self.ells[0] <= self.lcut <= self.ells[-1]:
            raise ValueError("lcut must be within the configured ell axis.")
        if self.n_bootstrap < 2:
            raise ValueError("n_bootstrap must be >= 2.")
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must be between 0 and 1.")
        self.ranges = {
            "full": np.ones(len(self.ells), dtype=bool),
            "common": self.ells <= self.lcut,
        }

    def execute(self) -> None:
        for split in self.splits:
            for epoch in self.model_epochs:
                with self.name_tracker.set_contexts(
                    dict(split=split.name, epoch=epoch)
                ):
                    self.process(split)
            logger.info("Done uncertainty metrics for %s", split.name)

    def process(self, split: Split) -> None:
        pred = np.asarray(self.in_pred.read(), dtype=float)
        thry = np.asarray(self.in_thry.read(), dtype=float)
        mean_thry = np.asarray(self.in_mean_thry.read(), dtype=float)

        expected = (split.n_sims, len(self.ells))
        if pred.shape != expected or thry.shape != expected:
            raise ValueError(
                f"Expected paired bundles of shape {expected}; "
                f"got pred={pred.shape}, thry={thry.shape}."
            )
        if mean_thry.shape != (len(self.ells),):
            raise ValueError(
                f"MeanTheory shape mismatch: {mean_thry.shape}."
            )

        ranges = {
            "full": [int(self.ells[0]), int(self.ells[-1])],
            "common": [int(self.ells[0]), int(self.lcut)],
        }
        results = {
            "ranges": ranges,
            "n_simulations": int(pred.shape[0]),
            "n_bootstrap": self.n_bootstrap,
            "seed": self.seed,
            "confidence": self.confidence,
            "standard_deviation": {},
            "bootstrap": {},
        }

        # SD of per-simulation scalar errors, not error bars on mean estimates.
        residual = pred - thry
        with np.errstate(divide="ignore", invalid="ignore"):
            relative = np.divide(
                residual, thry,
                out=np.full_like(residual, np.nan),
                where=(thry != 0),
            )

        for name, mask in self.ranges.items():
            sim_mse = np.mean(residual[:, mask] ** 2, axis=1)
            sim_mare = np.mean(np.abs(relative[:, mask]), axis=1) * 100
            results["standard_deviation"][name] = {
                "mse": float(np.std(sim_mse, ddof=1)),
                "abs_rel_err_pct": float(np.std(sim_mare, ddof=1)),
            }

        # Seed reset for each method. Equal split sizes and simulation ordering
        # then give identical index sequences for paired comparisons.
        rng = np.random.default_rng(self.seed)
        draws = {name: {} for name in self.ranges}
        for _ in range(self.n_bootstrap):
            idx = rng.integers(0, pred.shape[0], size=pred.shape[0])
            sample = compute_metrics(
                pred[idx], thry[idx], ells=self.ells,
                mean_theory=mean_thry,
            )
            for name, mask in self.ranges.items():
                scalar = GetScalarMetricsExecutor.compute(sample, mask)
                for metric, value in scalar.items():
                    draws[name].setdefault(metric, []).append(float(value))

        alpha = (1 - self.confidence) / 2
        for name, metric_draws in draws.items():
            results["bootstrap"][name] = {}
            for metric, observations in metric_draws.items():
                vals = np.asarray(observations)
                n_valid = int(np.isfinite(vals).sum())
                # Do not silently discard undefined bootstrap replications.
                if n_valid == len(vals):
                    lower, upper = np.quantile(vals, [alpha, 1 - alpha])
                else:
                    lower, upper = np.nan, np.nan
                results["bootstrap"][name][metric] = {
                    "lower": float(lower),
                    "upper": float(upper),
                    "n_valid": n_valid,
                }

        self.out_uncertainty.write(data=results)
