from typing import Dict, List
from pathlib import Path
import logging

from copy import deepcopy
import numpy as np
from omegaconf import DictConfig, OmegaConf

from cmbml.sims.get_wmap_params import get_wmap_indices, pull_params_from_file

from cmbml.sims.random_seed_manager import SeedFactory
from cmbml.core.asset_handlers import Config
from cmbml.core import (
    BaseStageExecutor,
    Split,
    Asset,
    AssetWithPathAlts
)


logger = logging.getLogger(__name__)


class ForegroundConfigExecutor(BaseStageExecutor):
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
        super().__init__(cfg, stage_str="make_fg_configs")

        self.out_fg_config: Asset = self.assets_out['fg_config']
        out_fg_config_handler: Config

        self.fg_seed_factory = SeedFactory(cfg.model.sim.fg_seed_template)
        self.fg_ranges = OmegaConf.to_container(cfg.model.sim.fgs, resolve=True)

    def execute(self) -> None:
        """
        Executes the configuration generation process for all splits and sims.
        """
        logger.debug(f"Running {self.__class__.__name__} execute() method.")
        for split in self.splits:
            with self.name_tracker.set_context("split", split.name):
                self.process_split(split)

    def process_split(self, split: Split) -> None:
        """
        Processes the given split.

        Args:
            split (Split): The split to process.
        """
        for sim in split.iter_sims():
            with self.name_tracker.set_context("sim_num", sim):
                self.process_sim(split, sim)

    def process_sim(self, split, sim):
        settings = {}
        fg_distributions = deepcopy(self.fg_ranges)
        for fg, fg_settings in fg_distributions.items():
            settings[fg] = {}
            for top_param, dist in fg_settings.items():
                if top_param != "dist":
                    continue
                for param, param_settings in dist.items():
                    seed = self.fg_seed_factory.get_seed(split=split.name, sim=sim, fg_str=fg, fg_param=param)
                    rng = np.random.default_rng(seed)
                    distribution = param_settings.pop("draw")
                    unit = None
                    if "unit" in param_settings:
                        unit = param_settings.pop("unit")
                    if distribution == "Uniform":
                        use_class = rng.uniform
                    elif distribution == "Normal":
                        use_class = rng.normal
                    else:
                        raise NotImplementedError("Only 'Uniform' and 'Normal' are currently implemented")
                    draw = use_class(**param_settings)
                    if unit is not None:
                        settings[fg][param] = {"value": draw, "unit": unit}
                    else:
                        settings[fg][param] = {"value": draw}
        self.out_fg_config.write(data=settings)
