from typing import Dict, List, Union, Any
from pathlib import Path
from pprint import pprint

import numpy as np

from get_wmap_params import get_indices, pull_params_from_file


class CosmologicalParameters:
    wmap_to_camp_dict = {'H0': 'H0',
                         'omegach2': 'omch2',
                         'omegabh2': 'ombh2',
                         'ns002': 'ns',
                         'tau': 'tau'}

    def __init__(self, params_dict: Dict[str, Any]) -> None:
        self.params = params_dict

    def camb_format(self):
        res_dict = {}
        for k, v in self.params.items():
            try:
                res_dict[CosmologicalParameters.wmap_to_camp_dict[k]] = v
            except KeyError as e:
                if k == "chain_idx": pass
                else: raise e
        return res_dict


class CosmologicalParameterBatch(CosmologicalParameters):
    def __init__(self, params_dict: Dict[str, List[Union[int, float]]]) -> None:
        super().__init__(params_dict)

    def get_idx(self, idx):
        return CosmologicalParameters({k: v[idx] for k, v in self.params.items()})


def get_params_from_wmap_chains(wmap_path, n_vals, params_to_get, rng: np.random.Generator):
    chain_idcs = get_indices(n_vals, rng)
    params_dict = pull_params_from_file(wmap_path,
                                        chain_idcs,
                                        params_to_get)
    return CosmologicalParameterBatch(params_dict)


def try_get_wmap_params(output=False):
    wmap_path = Path("wmap_assets/wmap_lcdm_wmap9_chains_v5")
    params_to_get = ['H0', 'omegach2', 'omegabh2', 'ns002', 'tau']
    n_vals = 2
    rng = np.random.default_rng(seed=0)
    res = get_params_from_wmap_chains(wmap_path, n_vals, params_to_get, rng)
    if output:
        pprint(res.params)
        pprint(res.get_idx(1).params)
        # TODO some sort of assert to check that the values actually match for all the keys
    return res


def try_get_wmap_params_with_camb_keys(output=False):
    res = try_get_wmap_params()
    single_param_set = res.get_idx(1)
    if output:
        pprint(single_param_set.params)
        pprint(single_param_set.camb_format())


if __name__ == "__main__":
    # try_get_wmap_params(True)
    try_get_wmap_params_with_camb_keys(True)
