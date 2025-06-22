from typing import Dict, List
from pathlib import Path
import logging

import numpy as np
from omegaconf import DictConfig, OmegaConf

from cmbml.core.asset_handlers import Config
from cmbml.core import (
    BaseStageExecutor,
    Split,
    Asset,
    AssetWithPathAlts
)
from cmbml.sims.random_seed_manager import SeedFactory


logger = logging.getLogger(__name__)


class ParamConfigExecutor(BaseStageExecutor):
    """
    ConfigExecutor is responsible for generating the configuration files for the simulation.

    Attributes:
        out_split_config (Asset): The output asset for the split configuration.
        out_wmap_config (AssetWithPathAlts): The output asset for the WMAP configuration.
        wmap_param_labels (List[str]): The labels for the WMAP parameters.
        wmap_chain_length (int): The length of the WMAP chains.
        wmap_chains_dir (Path): The directory containing the WMAP chains.
        seed (int): The seed for the WMAP indices.

    Methods:
        execute() -> None:
            Executes the configuration generation process.
        process_split(split: Split, these_idces) -> None:
            Processes the given split with the given WMAP indices.
        n_ps_for_split(split: Split) -> int:
            Determines the number of power spectra for the given split.
        make_chain_idcs_for_each_split(seed: int) -> Dict[str, List[int]]:
            Generates the WMAP chain indices for each split.
        make_cosmo_param_configs(chain_idcs, split) -> None:
            Generates the cosmological parameter configurations for the given chain indices
    """
    def __init__(self, cfg: DictConfig) -> None:
        # The following stage_str must match the pipeline yaml
        super().__init__(cfg, stage_str="make_sim_configs")

        self.out_wmap_config: AssetWithPathAlts = self.assets_out['cosmo_config']
        out_wmap_config_handler: Config

        self.seed_template = cfg.model.sim.cmb.seed_template
        self.params = cfg.model.sim.cmb.camb_params
        self.seed_factory = SeedFactory(self.seed_template)

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute() method.")
        for split in self.splits:
            with self.name_tracker.set_context("split", split.name):
                self.process_split(split)

    def process_split(self, split: Split) -> None:
        ps_fidu_fixed = split.ps_fidu_fixed

        if ps_fidu_fixed:
            these_params = self.get_cosmo_params(split, "fixed")
            self.out_wmap_config.write(use_alt_path=True, data=these_params)
            return
        
        for sim in split.iter_sims():
            with self.name_tracker.set_context("sim_num", sim):
                these_params = self.get_cosmo_params(split, sim)
                self.out_wmap_config.write(use_alt_path=False, data=these_params)

    def get_cosmo_params(self, split, sim) -> Dict[str, List[float]]:
        seed = self.seed_factory.get_seed(
            split=split.name,
            sim=sim,
        )

        rng = np.random.default_rng(seed)
        param_draws = {}
        for key, values in self.params.items():
            if key == "ln1010as":
                ln1010As = rng.normal(values["mean"], values["std"])
                As = np.exp(ln1010As)*1e-10
                param_draws["As"] = As
            elif "value" in values:
                # If the parameter has a fixed value, use that
                param_draws[key] = values["value"]
            else:
                param_draws[key] = rng.normal(values["mean"], values["std"])
        return param_draws
