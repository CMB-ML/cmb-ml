"""
For use with comparison methods.
"""
import logging

from omegaconf import DictConfig

from cmbml.core import Asset, BaseStageExecutor
from cmbml.utils.parse_ell_list import parse_ells
import cmbml.utils.degree_selection as ds


logger = logging.getLogger(__name__)


class GetPolyDegreeExecutor(BaseStageExecutor):
    """
    Select the optimal polynomial smoothing degree using validation spectra.
    """
    def __init__(self, cfg: DictConfig, stage_str="find_sm_degree") -> None:
        super().__init__(cfg, stage_str)
        self.out_deg_cfg: Asset = self.assets_out["deg_cfg"]
        self.in_pred: Asset = self.assets_in["ps_pred_bundle"]
        self.in_thry: Asset = self.assets_in["ps_thry_bundle"]

        # Per model details
        self.cleaning_model = cfg.model.cleaning.cleaning_model
        self.do_smooth = cfg.model.cleaning.do_smooth
        self.score_ell_max = cfg.model.cleaning.get("sm_score_ell_max", None)
        self.sm_epoch = cfg.model.cleaning.sm_epoch

        # Other settings
        self.use_ells = cfg.model.analysis.use_ells
        self.ells = parse_ells(self.use_ells)

        self.sm_basis_kind = cfg.get("sm_basis_kind", None)
        if self.sm_basis_kind is None:
            msg = f"No smoothing choice specified."
            logger.warning(msg)
            raise ValueError(msg)
        self.deg_range = cfg.get("sm_deg_range", None)
        if self.deg_range is None:
            msg = f"No degree range specified."
            logger.warning(msg)
            raise ValueError(msg)
        elif len(self.deg_range) != 2:
            msg = f"Degree range must be a pair of values (min, max). Got {self.deg_range}"
            logger.warning(msg)
            raise ValueError(msg)
        elif self.deg_range[0] > self.deg_range[1]:
            msg = f"Degree range min > max. Got {self.deg_range}"
            logger.warning(msg)
            raise ValueError(msg)
        elif self.deg_range[0] < 1:
            msg = f"Degree range min must be >=1. Got {self.deg_range}"
            logger.warning(msg)
            raise ValueError(msg)
        self.criterion = cfg.get("sm_criterion", None)
        if self.criterion is None:
            msg = f"No criterion specified."
            logger.warning(msg)
            raise ValueError(msg)

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        try:
            split = self.splits['Valid']
        except KeyError:
            raise ValueError(f"Getting the best polynomial degree is locked to Validation data.")

        lo, hi = self.deg_range

        model = self.cleaning_model
        if self.do_smooth:
            context = dict(epoch=self.sm_epoch, split=split.name)
            with self.name_tracker.set_contexts(context):
                pred = self.in_pred.read()
                thry = self.in_thry.read()
            sweeps = ds.sweep_smoothing(
                            self.ells, pred, thry,
                            smoother=self.sm_basis_kind,
                            model=model,
                            split=split.name,
                            degrees=range(lo, hi + 1))
            sel = ds.select_from_sweeps(
                {model: sweeps},
                criterion=self.criterion,
                ell_max={model: self.score_ell_max},
                use_ells=self.use_ells,
                epoch=self.sm_epoch
            )
        else:
            sel = ds.DegreeSelection(degrees={model: None}, 
                                     score_curves={model: {}},
                                     use_ells=self.use_ells,
                                     score_ell_max={model: None},
                                     criterion=None,
                                     smoother=None,
                                     selection_split=split.name, 
                                     epoch=self.sm_epoch
                                     )    
        self.out_deg_cfg.write(data=sel.to_dict())
