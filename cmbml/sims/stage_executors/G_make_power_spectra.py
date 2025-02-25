import logging

from omegaconf import DictConfig
from tqdm import tqdm

from cmbml.core import (
    BaseStageExecutor,
    Split,
    AssetWithPathAlts
)

from cmbml.sims.physics_cmb import make_camb_ps

from cmbml.core.asset_handlers.psmaker_handler import CambPowerSpectrum # Import to register handler
from cmbml.core.asset_handlers import Config


logger = logging.getLogger(__name__)


class TheoryPSExecutor(BaseStageExecutor):
    """
    TheoryPSExecutor is responsible for generating the theoretical power spectra for a given simulation scenario.

    Attributes:
        out_cmb_ps (AssetWithPathAlts): The output asset for the CMB power spectra.
        in_wmap_config (AssetWithPathAlts): The input asset for the WMAP configuration.
        max_ell_for_camb (int): The maximum ell value for the CAMB power spectrum calculation.
        wmap_param_labels (List[str]): The labels for the WMAP parameters.
        camb_param_labels (List[str]): The labels for the CAMB parameters.
    
    Methods:
        execute() -> None:
            Executes the theoretical power spectrum generation process.
        process_split(split: Split) -> None:
            Processes the given split for the theoretical power spectrum generation.
        make_ps(wmap_params: AssetWithPathAlts, ps_asset: AssetWithPathAlts, use_alt_path: bool) -> None:
            Generates the theoretical power spectra for the given WMAP parameters and writes them to the output asset.
    """

    def __init__(self, cfg: DictConfig) -> None:
        # The following stage_str must match the pipeline yaml
        super().__init__(cfg, stage_str='make_theory_ps')

        self.max_ell_for_camb = cfg.model.sim.cmb.ell_max
        self.wmap_param_labels = cfg.model.sim.cmb.wmap_params
        self.camb_param_labels = cfg.model.sim.cmb.camb_params_equiv

        self.out_cmb_ps: AssetWithPathAlts = self.assets_out['cmb_ps']
        self.in_wmap_config: AssetWithPathAlts = self.assets_in['wmap_config']

        out_cmb_ps_handler: CambPowerSpectrum
        in_wmap_config_handler: Config

    def execute(self) -> None:
        """
        Executes the theoretical power spectrum generation process for all splits and sims.
        """
        logger.debug(f"Running {self.__class__.__name__} execute() method.")
        self.default_execute()  # In BaseStageExecutor

    def process_split(self, split: Split) -> None:
        """
        Processes all sims for a split, making theory power spectra.

        Args:
            split (Split): The split to process.
        """
        if split.ps_fidu_fixed:
            self.make_ps(self.in_wmap_config, self.out_cmb_ps, use_alt_path=True)
        else:
            for sim in tqdm(split.iter_sims()):
                with self.name_tracker.set_context("sim_num", sim):
                    self.make_ps(self.in_wmap_config, self.out_cmb_ps, use_alt_path=False)

    def make_ps(self, 
                wmap_params: AssetWithPathAlts, 
                ps_asset: AssetWithPathAlts,
                use_alt_path) -> None:
        """
        Generates the theoretical power spectra for the given WMAP parameters and writes them to the output asset.

        Args:
            wmap_params (AssetWithPathAlts): The WMAP parameters asset.
            ps_asset (AssetWithPathAlts): The output asset for the power spectra.
            use_alt_path (bool): If using a single power spectrum for the split, 
                                 it is written to a different location.
        """
        # Pull cosmological parameters from wmap_configs created earlier
        cosmo_params = wmap_params.read(use_alt_path=use_alt_path)
        # cosmological parameters from WMAP chains have (slightly) different names in camb
        cosmo_params = self._translate_params_keys(cosmo_params)

        camb_results = make_camb_ps(cosmo_params, lmax=self.max_ell_for_camb)
        ps_asset.write(use_alt_path=use_alt_path, data=camb_results, lmax=self.max_ell_for_camb)

    def _translate_params_keys(self, src_params):
        translation_dict = self._param_translation_dict()
        target_dict = {}
        for k in src_params:
            if k == "chain_idx":
                continue
            target_dict[translation_dict[k]] = src_params[k]
        return target_dict

    def _param_translation_dict(self):
        translation = {}
        for i in range(len(self.wmap_param_labels)):
            translation[self.wmap_param_labels[i]] = self.camb_param_labels[i]
        return translation
