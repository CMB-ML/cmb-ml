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
        self.cosmo_params = cfg.model.sim.cmb.camb_params

        self.out_cmb_ps: AssetWithPathAlts = self.assets_out['cmb_ps']
        self.in_cosmo_config: AssetWithPathAlts = self.assets_in['cosmo_config']

        self.need_xl = cfg.model.sim.cmb.get('use_chains', False)

        out_cmb_ps_handler: CambPowerSpectrum
        in_cosmo_config_handler: Config

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
            for _ in tqdm(range(1)):
                self.make_ps(self.in_cosmo_config, self.out_cmb_ps, use_alt_path=True)
        else:
            for sim in tqdm(split.iter_sims()):
                with self.name_tracker.set_context("sim_num", sim):
                    self.make_ps(self.in_cosmo_config, self.out_cmb_ps, use_alt_path=False)

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
        if self.need_xl:
            cosmo_params = self._translate_params_keys(cosmo_params)

        camb_results = make_camb_ps(cosmo_params, lmax=self.max_ell_for_camb)
        ps_asset.write(use_alt_path=use_alt_path, data=camb_results, lmax=self.max_ell_for_camb)

    def _translate_params_keys(self, src_params):
        out_params = {}
        for param_k in self.cosmo_params.keys():
            # if param_k not in self.cosmo_params:
            #     raise ValueError(f"Key {param_k} not in {self.cosmo_params}. Was this config written in this pipeline?")
            xl_dict = self.cosmo_params[param_k]
            if not xl_dict:  # If grabbing other elements of the chain, but not using them in CAMB
                continue
            # The key for use with CAMB is in the config under 'camb'
            new_k = xl_dict['camb']

            if param_k not in src_params:
                if 'value' in xl_dict:  # We force a value for this parameter (pivot_scalar, for example)
                    new_v = xl_dict['value']
                else:
                    raise ValueError(f"Key {param_k} not in source parameters {src_params}. "
                                     f"Was this config written in this pipeline?")
            else:
                new_v = src_params[param_k]
                if 'factor' in xl_dict:
                    new_v = new_v * xl_dict['factor']
            out_params[new_k] = new_v
        return out_params

        # for param_k, param_v in src_params.items():
        #     if param_k not in self.cosmo_params:
        #         raise ValueError(f"Key {param_k} not in {self.cosmo_params}. Was this config written in this pipeline?")
        #     xl_dict = self.cosmo_params[param_k]
        #     if not xl_dict:  # If grabbing other elements of the chain, but not using them in CAMB
        #         continue
        #     new_k = xl_dict['camb']
        #     new_v = param_v
        #     if 'factor' in xl_dict:
        #         new_v = param_v * xl_dict['factor']
        #     if 'value' in xl_dict:
        #         new_v = xl_dict['value']
        #     out_params[new_k] = new_v
        # return out_params
