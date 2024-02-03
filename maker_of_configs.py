from typing import List, Dict
from omegaconf import DictConfig, OmegaConf

from namer_dataset_output import DatasetFilesNamer
from namer_wmap import WMAPFilesNamer
from get_wmap_params import get_wmap_indices, pull_params_from_file


class DatasetConfigsMaker:
    def __init__(self, 
                 conf: DictConfig):
        self.dsf = DatasetFilesNamer(conf)
        self.wmap_files = WMAPFilesNamer(conf)
        self.wmap_param_labels = conf.simulation.cmb.wmap_params

    def setup_folders(self):
        # Ensure correct filesystem before creating folders in strange places
        self.dsf.assume_dataset_root_exists()
        for split in self.dsf.iter_splits():
            for sim in split.iter_sims():
                sim.make_folder()

    def make_chain_idcs_for_each_split(self, seed:int) -> Dict[str, List[int]]:
        n_indices_total = self.dsf.total_n_ps
        all_chain_indices = get_wmap_indices(n_indices_total, seed)
        # convert from numpy array of np.int64 to List[int] for OmegaConf
        all_chain_indices = getattr(all_chain_indices, "tolist", lambda: all_chain_indices)()
        
        last_index_used = 0
        chain_idcs_dict = {}
        for split in self.dsf.iter_splits():
            first_index = last_index_used
            last_index_used = first_index + split.n_ps
            chain_idcs_dict[split.name] = all_chain_indices[first_index: last_index_used]

        return chain_idcs_dict

    def make_split_configs(self, chain_idcs_dict):
        for split in self.dsf.iter_splits():
            split_cfg_dict = dict(
                ps_fidu_fixed = split.ps_fidu_fixed,
                n_sims = split.n_sims,
                wmap_chain_idcs = chain_idcs_dict[split.name]
            )

            split_yaml = OmegaConf.create(split_cfg_dict)
            split.write_split_conf_file(split_yaml)

    def make_cosmo_param_configs(self):
        for split in self.dsf.iter_splits():
            split_conf = split.read_split_conf_file()
            wmap_params = pull_params_from_file(wmap_chain_path=self.wmap_files.wmap_chains_dir,
                                                chain_idcs=split_conf.wmap_chain_idcs,
                                                params_to_get=self.wmap_param_labels)

            if split.ps_fidu_fixed:
                n_sims_to_process = 1
            else:
                n_sims_to_process = split.n_sims
            
            for i in range(n_sims_to_process):
                these_params = {key: values[i] for key, values in wmap_params.items()}
                sim = split.get_sim(i)
                sim.write_wmap_params_file(these_params)
