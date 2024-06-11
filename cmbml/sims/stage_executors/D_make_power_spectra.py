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
from cmbml.core.asset_handlers.asset_handlers_base import Config


logger = logging.getLogger(__name__)


class TheoryPSExecutor(BaseStageExecutor):
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
        logger.debug(f"Running {self.__class__.__name__} execute() method.")
        self.default_execute()  # In BaseStageExecutor

    def process_split(self, split: Split) -> None:
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
        # Pull cosmological parameters from wmap_configs created earlier
        cosmo_params = wmap_params.read(use_alt_path=use_alt_path)
        # cosmological parameters from WMAP chains have (slightly) different names in camb
        cosmo_params = self._translate_params_keys(cosmo_params)

        camb_results = make_camb_ps(cosmo_params, lmax=self.max_ell_for_camb)
        ps_asset.write(use_alt_path=use_alt_path, data=camb_results)

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
