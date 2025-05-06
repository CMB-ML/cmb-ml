from typing import Dict, List
from pathlib import Path
import logging

from omegaconf import DictConfig, OmegaConf

from cmbml.sims.get_wmap_params import get_wmap_indices, pull_params_from_file

from cmbml.core.asset_handlers import Config
from cmbml.core import (
    BaseStageExecutor,
    Split,
    Asset,
    AssetWithPathAlts
)


logger = logging.getLogger(__name__)


class ConfigExecutor(BaseStageExecutor):
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

        self.in_wmap_chains: Asset = self.assets_in['wmap_chains']

        self.out_split_config: Asset = self.assets_out['split_configs']
        self.out_wmap_config: AssetWithPathAlts = self.assets_out['wmap_config']
        out_split_config_handler: Config
        out_wmap_config_handler: Config

        self.wmap_param_labels = self.get_param_labels(cfg.model.sim.cmb.wmap_camb_params)
        self.wmap_chain_length = cfg.model.sim.cmb.wmap_chain_length

        self.seed = cfg.model.sim.cmb.wmap_indcs_seed

    def get_param_labels(self, param_cfg: DictConfig) -> None:
        """
        Get the parameter labels for the WMAP parameters.

        Args:
            param_cfg (DictConfig): The configuration for the parameters.
        """
        param_cfg = OmegaConf.to_container(param_cfg, resolve=True)
        param_labels = list(param_cfg.keys())
        param_labels.remove("chain_idx")
        return param_labels

    def execute(self) -> None:
        """
        Executes the configuration generation process for all splits and sims.
        """
        logger.debug(f"Running {self.__class__.__name__} execute() method.")
        all_idices = self.make_chain_idcs_for_each_split(self.seed)
        for split in self.splits:
            with self.name_tracker.set_context("split", split.name):
                self.process_split(split, all_idices[split.name])

    def process_split(self, split: Split, these_idces) -> None:
        """
        Processes the given split with the given WMAP indices.

        Args:
            split (Split): The split to process.
            these_idces (List[int]): The rows of the WMAP chain for this split.
        """
        split_cfg_dict = dict(
            ps_fidu_fixed = split.ps_fidu_fixed,
            n_sims = split.n_sims,
            wmap_chain_idcs = these_idces
        )

        with self.name_tracker.set_context("split", split.name):
            self.out_split_config.write(data=split_cfg_dict)

        self.make_cosmo_param_configs(split_cfg_dict['wmap_chain_idcs'], split)

    @staticmethod
    def n_ps_for_split(split: Split):
        return 1 if split.ps_fidu_fixed else split.n_sims

    def make_chain_idcs_for_each_split(self, seed:int) -> Dict[str, List[int]]:
        """
        Compile a list of distinct indices for each split. Structured this way to 
        produce sets of WMAP parameters which are distinct across all splits.

        Args:
            seed (int): The seed to use for generation.

        Returns:
            Dict: A dictionary where the keys are the split names and the values
            are Lists of the chain indices.
        """
        # Some splits will have only one power spectrum, others will have one for each simulation.
        #   Count them:
        n_indices_total = 0
        for split in self.splits:
            n_indices_total += self.n_ps_for_split(split)

        # For each power spectrum to be generated, we'll need a set of WMAP parameters.
        all_chain_indices = get_wmap_indices(n_indices_total, seed, wmap_chain_length=self.wmap_chain_length)
        
        # Convert from numpy array of np.int64 to List[int] for OmegaConf
        all_chain_indices = getattr(all_chain_indices, "tolist", lambda: all_chain_indices)()
        
        # Give the appropriate number of indices to each split
        last_index_used = 0
        chain_idcs_dict = {}
        for split in self.splits:
            first_index = last_index_used
            last_index_used = first_index + self.n_ps_for_split(split)
            chain_idcs_dict[split.name] = all_chain_indices[first_index: last_index_used]

        return chain_idcs_dict

    def make_cosmo_param_configs(self, chain_idcs, split):
        """
        Make the cosmological parameter configurations for the given chain indices.

        Args:
            chain_idcs (List[int]): The indices of the WMAP chain to use.
            split (Split): The split for which to make the configurations.
        """
        wmap_chains_dir = self.in_wmap_chains.path
        wmap_params = pull_params_from_file(wmap_chain_path=wmap_chains_dir,
                                            chain_idcs=chain_idcs,
                                            params_to_get=self.wmap_param_labels,
                                            wmap_chain_length=self.wmap_chain_length)

        if split.ps_fidu_fixed:
            these_params = {key: values[0] for key, values in wmap_params.items()}
            self.out_wmap_config.write(use_alt_path=True, data=these_params)
        else:
            for i in split.iter_sims():
                these_params = {key: values[i] for key, values in wmap_params.items()}
                with self.name_tracker.set_context("sim_num", i):
                    self.out_wmap_config.write(use_alt_path=False, data=these_params)
