import logging

import numpy as np
from omegaconf import DictConfig

from cmbml.core import Split, Asset, BaseStageExecutor
from cmbml.utils.parse_ell_list import parse_ells
import cmbml.utils.degree_selection as ds
import cmbml.utils.smoothing as sm


logger = logging.getLogger(__name__)


class BundleSmoothExecutor(BaseStageExecutor):
    """
    Apply polynomial smoothing to bundled power spectra.
    """
    def __init__(self, cfg: DictConfig, stage_str="bundle_smooth") -> None:
        super().__init__(cfg, stage_str)
        self.out_ps: Asset = self.assets_out["ps_bundle"]
        self.in_ps: Asset = self.assets_in["pred"]
        self.in_deg_cfg: Asset = self.assets_in["deg_cfg"]

        self.cleaning_model = cfg.model.cleaning.cleaning_model
        self.do_smooth = cfg.model.cleaning.do_smooth

        self.use_ells = cfg.model.analysis.use_ells
        self.ells = parse_ells(self.use_ells)

        self.sm_basis_kind = cfg.get("sm_basis_kind", None)
        self.smoother = None

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")

        self.set_up_smoother()

        for split in self.splits:
            for epoch in self.model_epochs:
                context = dict(split=split.name, epoch=epoch)
                with self.name_tracker.set_contexts(context):
                    self.process(split)
            logger.info(f"Done bundling {split.name} spectra.")

    def set_up_smoother(self):
        deg_cfg = self.in_deg_cfg.read()
        sel = ds.DegreeSelection.from_dict(deg_cfg)
        degree = sel.degree_for(self.cleaning_model)

        if degree is None:                    # unsmoothed model (obs2ps): nothing to check
            if self.do_smooth:
                raise ValueError("Smoothing degree not found, but Hydra says to smooth.")
                # This could happen if the pipeline is run out of order or the saved file
                #   has been modified.
            self.smoother = None
            return
        if not self.do_smooth:
            raise ValueError(f"Config says not to smooth, but the artifact carries "
                             f"degree {degree} for {self.cleaning_model!r}.")
        
        if sel.use_ells != self.use_ells:
            raise ValueError(f"degree chosen over ells {sel.use_ells!r}, applying over {self.use_ells!r}")
        if sel.smoother != self.sm_basis_kind:
            raise ValueError(f"degree chosen with {sel.smoother!r}, applying with {self.sm_basis_kind!r}")

        self.smoother = sm.make_smoother(sel.smoother, degree=degree)

    def process(self, split: Split) -> None:
        pred = self.in_ps.read()
        pred = np.atleast_2d(pred)

        if self.smoother is None:
            self.out_ps.write(data=pred)
            return

        # Reset counter
        self.smoother.ill_conditioned_count = 0
        result = sm.smooth_ensemble(self.smoother, self.ells, pred)
        if getattr(self.smoother, "ill_conditioned_count", 0):
            logger.warning(f"{self.smoother.ill_conditioned_count} ill-conditioned fits at degree {self.smoother.degree}")
        self.out_ps.write(data=result)


class BundleSmoothTrueExecutor(BundleSmoothExecutor):
    def __init__(self, cfg: DictConfig, stage_str="bundle_smooth_true") -> None:
        super().__init__(cfg, stage_str)
