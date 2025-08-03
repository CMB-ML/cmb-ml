import numpy as np
import healpy as hp
from pysm3.units import Quantity


def inpaint_with_neighbor_mean(some_map, max_iters=10):
    """
    Inpaint a HealpyMap by replacing UNSEEN pixels with the mean of their neighbors.
    
    Parameters
    ----------
    some_map : np.ndarray
        The input map to inpaint.
    max_iters : int, optional
        Maximum number of iterations to perform (default is 10).
    
    Returns
    -------
    np.ndarray
        The inpainted map.
    """

    try:
        some_map_unit = some_map.unit  # Convert to numpy array if needed
        some_map = some_map.value
    except AttributeError:
        some_map_unit = None

    nside = hp.get_nside(some_map)
    for iter in range(max_iters):
        unseen_mask = (some_map == hp.UNSEEN)
        if not np.any(unseen_mask):
            break  # No more unseen pixels to process

        for px in np.where(unseen_mask)[0]:
            theta, phi = hp.pix2ang(nside=nside, ipix=px)
            neigh = hp.get_all_neighbours(nside=nside, theta=theta, phi=phi)
            neigh_vals = some_map[neigh]
            valid_vals = neigh_vals[neigh_vals != hp.UNSEEN]
            if len(valid_vals) > 0:
                # print(f"Inpainting pixel {px} with neighbors: {valid_vals}")
                some_map[px] = np.nanmean(valid_vals)

        if iter == max_iters - 1:
            print(f"Warning: Maximum iterations ({max_iters}) reached without full inpainting.")

    if some_map_unit is not None:
        some_map = Quantity(some_map, some_map_unit)

    return some_map