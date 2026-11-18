"""
For use with comparison methods.
"""
import logging

from omegaconf import DictConfig

from cmbml.core import Asset, BaseStageExecutor, Split
from cmbml.core.asset_handlers import NpyHandler
from cmbml.utils.parse_ell_list import parse_ells


logger = logging.getLogger(__name__)


class MakeMeanSpectrumExecutor(BaseStageExecutor):
    """
    Calculate the mean theoretical power spectrum of the training dataset.
    """
    def __init__(self, cfg: DictConfig, stage_str="make_mean_spectrum") -> None:
        super().__init__(cfg, stage_str)
        self.out_mean_spectrum: Asset = self.assets_out["mean_thry"]
        out_mean_spectra_handler: NpyHandler

        self.in_thry: Asset = self.assets_in["ps_thry_bundle"]
        in_thry_bundle_handler: NpyHandler

        # Other settings
        self.use_ells = cfg.model.analysis.use_ells
        self.ells = parse_ells(self.use_ells)

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute()")
        try:
            split: Split = self.splits['Train']
        except KeyError:
            raise ValueError(f"Getting the mean spectrum is locked to Training data.")

        with self.name_tracker.set_context("split", split.name):
            thry = self.in_thry.read()

        if thry.ndim != 2:
            raise ValueError(
                f"Training bundle must be 2D (N_sim, N_ell), "
                f"got shape {thry.shape}."
            )

        load_N_sim, load_N_ell = thry.shape
        want_N_sim = split.n_sims
        want_N_ell = len(self.ells)

        if load_N_sim != want_N_sim:
            raise ValueError(
                f"Training bundle simulation count mismatch: "
                f"expected {want_N_sim}, got {load_N_sim}."
            )

        if load_N_ell != want_N_ell:
            raise ValueError(
                f"Training bundle multipole count mismatch: "
                f"expected {want_N_ell}, got {load_N_ell}."
            )

        logger.info(f"MeanSpectra to be written at {self.out_mean_spectrum.path}")
        self.out_mean_spectrum.write(data=thry.mean(axis=0))
