from typing import Dict, List
from pathlib import Path
import logging

from omegaconf import DictConfig, OmegaConf

from ..get_wmap_params import get_wmap_indices, pull_params_from_file

from ...core.asset_handlers import Config
from ...core import (
    BaseStageExecutor,
    Split,
    Asset,
    AssetWithPathAlts
)


logger = logging.getLogger(__name__)


class ConfigExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following stage_str must match the pipeline yaml
        super().__init__(cfg, stage_str="make_sim_configs")

        self.out_split_config: Asset = self.assets_out['split_configs']
        self.out_wmap_config: AssetWithPathAlts = self.assets_out['wmap_config']
        out_split_config_handler: Config
        out_wmap_config_handler: Config

        self.wmap_param_labels = cfg.model.sim.cmb.wmap_params
        self.wmap_chain_length = cfg.model.sim.cmb.wmap_chain_length
        self.wmap_chains_dir = Path(cfg.local_system.wmap_chains_dir)

        self.seed = cfg.model.sim.cmb.wmap_indcs_seed

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute() method.")
        all_idices = self.make_chain_idcs_for_each_split(self.seed)
        for split in self.splits:
            with self.name_tracker.set_context("split", split.name):
                self.process_split(split, all_idices[split.name])

    def process_split(self, split: Split, these_idces) -> None:
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
        # We want to generate a set of WMAP parameters for each power spectrum to be generated.
        # We ALSO want all of the sets to be different.
        # We first compile a list of indices so that we know they're distinct, then 
        #    portion them out to the splits.
        
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
        wmap_params = pull_params_from_file(wmap_chain_path=self.wmap_chains_dir,
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
