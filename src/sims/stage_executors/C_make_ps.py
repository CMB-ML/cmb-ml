from typing import Dict
import pysm3

import hydra
from omegaconf import DictConfig
from pathlib import Path

import pysm3

from ...core import (
    BaseStageExecutor,
    ExperimentParameters,
    Split,
    Asset
)

from ..physics_cmb import make_cmb_ps
from ...core.asset_handlers import _make_directories

class FidPSExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig, experiment: ExperimentParameters) -> None:
        self.stage_str = 'create-fid-ps'
        super().__init__(cfg, experiment)

        self.max_ell_for_camb = cfg.simulation.cmb.ell_max
        self.wmap_param_labels = cfg.simulation.cmb.wmap_params
        self.camb_param_labels = cfg.simulation.cmb.camb_params_equiv

        self.in_wmap_fixed: Asset = self.assets_in['wmap_config_fixed']
        self.in_wmap_varied: Asset = self.assets_in['wmap_config_varied']

        self.out_ps_fixed: Asset = self.assets_out['cmb_ps_fid_fixed']
        self.out_ps_varied: Asset = self.assets_out['cmb_ps_fid_varied']

    def execute(self) -> None:
        super().execute()
    
    def process_split(self, split: Split) -> None:
        if split.ps_fidu_fixed:
            path = self.out_ps_fixed.path
            self.make_ps(self.in_wmap_fixed, path)
        else:
            for sim in split.iter_sims():
                with self.name_tracker.set_context("sim_num", sim):
                    path = self.out_ps_varied.path
                    self.make_ps(self.in_wmap_varied, path)
    
    def make_ps(self, wmap_params: Asset, out_path: Path) -> None:
        cosmo_params = wmap_params.read()
        _make_directories(out_path)

        cosmo_params = self._translate_params_keys(cosmo_params)

        make_cmb_ps(cosmo_params, self.max_ell_for_camb, out_path)

    def _param_translation_dict(self):
        translation = {}
        for i in range(len(self.wmap_param_labels)):
            translation[self.wmap_param_labels[i]] = self.camb_param_labels[i]
        return translation
    
    def _translate_params_keys(self, src_params):
        translation_dict = self._param_translation_dict()
        target_dict = {}
        for k in src_params:
            if k == "chain_idx":
                continue
            target_dict[translation_dict[k]] = src_params[k]
        return target_dict