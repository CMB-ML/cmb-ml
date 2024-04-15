import os
import numpy as np


# ROWS_IN_CHAINS = 1296570

def pull_params_from_file(wmap_chain_path, chain_idcs, params_to_get, wmap_chain_length):
    """
    Get parameters from wmap chains.
    
    For a tiny bit of speed, we do a single pass through the data files.
    """
    n_chain_rows = wmap_chain_length
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


def get_wmap_indices(n_indcs, seed:int, wmap_chain_length: int):
    rng = np.random.default_rng(seed=seed)
    set_of_indices = set(rng.integers(low=1, high=wmap_chain_length, size=n_indcs, endpoint=True))
    while len(set_of_indices) != n_indcs:
        set_of_indices.add(rng.integers(low=1, high=wmap_chain_length, size=1, endpoint=True)[0])
    return [int(idx)for idx in set_of_indices]
