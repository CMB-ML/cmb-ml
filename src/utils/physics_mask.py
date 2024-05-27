import logging
import numpy as np
import healpy as hp


logger = logging.getLogger(__name__)


def downgrade_mask(mask_data, nside_out, threshold=0.9):
    nside_in = hp.get_map_size(mask_data)
    if nside_in == nside_out:
        logger.info(f"Mask resolution matches map resolution. In: {nside_in}, Out: {nside_out}. No action taken.")
        return mask_data
    elif nside_in < nside_out:
        logger.warning(f"Mask resolution is lower than map resolution. Consider scaling it externally. This is an unhandled case. Proceed with caution.")
    downgraded_mask = hp.ud_grade(mask_data, nside_out)
    mask = apply_threshold(downgraded_mask, threshold)
    return mask


def apply_threshold(mask, thresh=0.9):
    # Per Planck 2015 results:IX. Diffuse component separation: CMB maps
    #    When downscaling mask maps; threshold the downscaled map
    #    They use 0.9
    return np.where(mask<thresh, 0, 1)
