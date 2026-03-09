from typing import Dict, List
from pathlib import Path
import logging

import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from cmbml.core.asset_handlers import Config
from cmbml.core import (
    BaseStageExecutor,
    Split,
    Asset,
    AssetWithPathAlts
)
from cmbml.sims.random_seed_manager import SeedFactory
from cmbml.sims import TheoryPSExecutor
from cmbml.sims.physics_cmb import make_camb_ps


logger = logging.getLogger(__name__)


class NearParamConfigExecutor(BaseStageExecutor):
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
        super().__init__(cfg, stage_str="make_near_cosmo_configs")

        self.out_wmap_config: AssetWithPathAlts = self.assets_out['near_cosmo_config']
        self.out_flag: Asset = self.assets_out['quick_flag']
        out_wmap_config_handler: Config

        self.in_wmap_config: AssetWithPathAlts = self.assets_in['cosmo_config']
        in_wmap_config_handler: Config

        self.seed_template = cfg.model.sim.cmb.seed_template_ps
        self.params = cfg.model.sim.cmb.camb_params
        self.seed_factory = SeedFactory(self.seed_template)
        try:
            self.n_sigma_near_params = OmegaConf.to_container(cfg.n_sigma_near_params)
        except ValueError:  # Permit int or float values; one parameter will be chosen randomly
            self.n_sigma_near_params = cfg.n_sigma_near_params
        self.change_params = cfg.change_params  # Either all or 1
        self.fixed_jitter = cfg.fixed_jitter

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute() method.")
        for split in self.splits:
            with self.name_tracker.set_context("split", split.name):
                self.process_split(split)

    def process_split(self, split: Split) -> None:
        for sim in split.iter_sims():
            with self.name_tracker.set_context("sim_num", sim):
                these_params = self.get_cosmo_params(split)
                self.out_wmap_config.write(use_alt_path=False, data=these_params)

    def get_cosmo_params(self, split: Split) -> Dict[str, List[float]]:
        sim_name = self.name_tracker.sim_name()
        seed = self.seed_factory.get_seed(
            split=split.name,
            sim=sim_name,
        )

        rng = np.random.default_rng(seed)

        # Get original parameters
        old_params = self.in_wmap_config.read(use_alt_path=split.ps_fidu_fixed)

        # Get directions to change parameters
        change_dict = {}
        if isinstance(self.n_sigma_near_params, dict):
            change_dict = self.n_sigma_near_params
            for k in change_dict:
                if k not in old_params:
                    raise ValueError(f"Key {k} not in original config.")
            elligible_keys = [k for k,v in change_dict.items() if "value" not in v]
        else:
            elligible_keys = [k for k,v in self.params.items() if "value" not in v]
        
        if self.change_params == 1:
            change_param = rng.choice(elligible_keys)
            change_dict[change_param] = self.n_sigma_near_params
        else:
            change_dict = {k: self.n_sigma_near_params for k in elligible_keys}

        for k, v in change_dict.items():
            fixed_jitter = rng.uniform(1-self.fixed_jitter, 
                                       1+self.fixed_jitter) if split.ps_fidu_fixed else 1
            change_dict[k] = rng.choice([-1, 1]) * v * fixed_jitter
            context = dict(param=k, 
                           ud="u" if np.sign(change_dict[k]) > 0 else "d",
                           amount=f"{np.abs(change_dict[k]):.1f}"
                           )
            with self.name_tracker.set_contexts(context):
                self.out_flag.write(data="")

        # Create new parameters
        new_params = {}
        for key, param_dict in self.params.items():
            if key == "ln1010as":  # E.g., if Planck-like
                old_As = old_params.pop("As")  # Stored as As for use with CAMB
                ln1010As = np.log(old_As*1e10)
                old_params["ln1010as"] = ln1010As
            
            if "value" in param_dict:  # This is like pivot_scalar, to be left alone
                new_params[key] = param_dict["value"]
                continue

            if key in change_dict:
                new_params[key] = change_dict[key] * param_dict["std"] + old_params[key]
            else:
                new_params[key] = old_params[key]

        if "ln1010as" in new_params:
            ln1010As = new_params.pop("ln1010as")
            As = np.exp(ln1010As)*1e-10
            new_params["As"] = As
        return new_params


class NearTheoryPSExecutor(TheoryPSExecutor):
    def __init__(self, cfg: DictConfig, stage_str="make_near_theory_ps"):
        super().__init__(cfg, stage_str=stage_str)

    def process_split(self, split: Split) -> None:
        """
        Processes all sims for a split, making theory power spectra.

        Args:
            split (Split): The split to process.
        """
        for sim in tqdm(split.iter_sims()):
            with self.name_tracker.set_context("sim_num", sim):
                camb_results = self.make_ps(self.in_cosmo_config, use_alt_path=False)
                self.out_cmb_ps.write(data=camb_results, lmax=self.max_ell_for_camb)
