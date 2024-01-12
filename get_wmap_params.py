import os
from pathlib import Path
from pprint import pprint
import numpy as np


ROWS_IN_CHAINS = 1296570

def pull_params_from_file(wmap_chain_path, chain_idcs, params_to_get):
    """

    Get parameters from wmap chains.
    
    For a tiny bit of speed, we do a single pass through the data files.
    """
    n_chain_rows = ROWS_IN_CHAINS
    n_vals = len(chain_idcs)

    # chain_idcs are not ordered
    # create a list of tuples: (orig_pos, sort_idx)
    sorted_chain_idcs = list(enumerate(chain_idcs))
    # reorder that by chain index
    sorted_chain_idcs = sorted(sorted_chain_idcs, key=lambda x: x[1])
    
    param_vals = {}
    for param in params_to_get:
        p_filename = wmap_chain_path / param
        # Figure out how many bytes to offset for each value
        p_filesize = os.stat(p_filename).st_size
        row_size = int(p_filesize / n_chain_rows)
        # Initialize the list of values for this parameter
        param_vals[param] = [None for _ in range(n_vals)]
        with open(p_filename) as f:
            for orig_pos, chain_idx in sorted_chain_idcs:
                location_in_file = (chain_idx - 1) * row_size
                if location_in_file >= 0:
                    f.seek(location_in_file)
                random_line = f.read(row_size)
                # We need to get the second value, the first is just the wmap index
                random_draw = random_line.split()[1]
                # Store the drawn value at the position matching the random draws
                param_vals[param][orig_pos] = float(random_draw)
        for i in range(n_vals):
            assert param_vals[param][i] is not None, "Error while drawing values"
        if param == 'a002':
            param_vals[param] = [v / 1e9 for v in param_vals[param]]
    param_vals['chain_idx'] = chain_idcs
    return param_vals


def get_indices(n_indcs, rng: np.random.Generator):
    return rng.integers(low=1, high=ROWS_IN_CHAINS, size=n_indcs, endpoint=True)


def try_getting_wmap_params():
    wmap_path = Path("wmap_assets/wmap_lcdm_wmap9_chains_v5")
    chain_idcs = [1, 2, 3, 4, 5, ROWS_IN_CHAINS]
    params_to_get = ['H0', 'omegam', 'omegab', 'sigma8', 'ns002', 'tau', 'a002']
    res = pull_params_from_file(wmap_path, chain_idcs, params_to_get)
    pprint(res)


def try_get_indices():
    res = get_indices(10, np.random.default_rng(seed=0))
    print(res)


def try_getting_wmap_params2():
    wmap_path = Path("wmap_assets/wmap_lcdm_wmap9_chains_v5")
    chain_idcs = get_indices(10, np.random.default_rng(seed=0))
    params_to_get = ['H0', 'omegam', 'omegab', 'sigma8', 'ns002', 'tau', 'a002']
    res = pull_params_from_file(wmap_path, chain_idcs, params_to_get)
    pprint(res)


if __name__ == "__main__":
    # try_getting_wmap_params()
    # try_get_indices()
    try_getting_wmap_params2()
