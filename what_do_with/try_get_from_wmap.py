from pathlib import Path
from pprint import pprint

import numpy as np

from src.sims.get_wmap_params import (ROWS_IN_CHAINS, 
                             pull_params_from_file,
                             get_wmap_indices)


WMAP_CHAINS_DIR = "wmap_assets/wmap_chains"
WMAP_CHAIN_LENGTH = 1296570


def try_getting_wmap_params():
    wmap_path = Path(WMAP_CHAINS_DIR)
    chain_idcs = [1, 2, 3, 4, 5, ROWS_IN_CHAINS]
    params_to_get = ['H0', 'omegam', 'omegab', 'sigma8', 'ns002', 'tau', 'a002']
    res = pull_params_from_file(wmap_path, chain_idcs, params_to_get, wmap_chain_length=WMAP_CHAIN_LENGTH)
    pprint(res)


def try_get_indices():
    res = get_wmap_indices(10, np.random.default_rng(seed=0), wmap_chain_length=WMAP_CHAIN_LENGTH)
    print(res)


def try_getting_wmap_params2():
    wmap_path = Path(WMAP_CHAINS_DIR)
    chain_idcs = get_wmap_indices(10, np.random.default_rng(seed=0), wmap_chain_length=WMAP_CHAIN_LENGTH)
    params_to_get = ['H0', 'omegam', 'omegab', 'sigma8', 'ns002', 'tau', 'a002']
    res = pull_params_from_file(wmap_path, chain_idcs, params_to_get, wmap_chain_length=WMAP_CHAIN_LENGTH)
    pprint(res)


def main():
    try_getting_wmap_params()
    try_get_indices()
    try_getting_wmap_params2()


if __name__=="__main__":
    main()
