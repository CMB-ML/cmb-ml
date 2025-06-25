import logging
import numpy as np
import healpy as hp


logger = logging.getLogger(__name__)


def convert_mask_from_Quantity(mask):
    try:
        mask = mask.value   # HealpyMap returns a Quantity
    except AttributeError:  # Mask is not a Quantity
        pass
    return mask


def downgrade_mask(mask_data, nside_out, threshold):
    """
    Downgrade the resolution of a mask to a specified resolution.

    Args:
        mask_data (np.ndarray): Numpy array representing the input mask data.
        nside_out (int): The desired output resolution.
        threshold (float): The threshold to apply to the downgraded mask.

    Returns:
        np.ndarray: The downgraded mask with the applied threshold.
    """
    mask_data = convert_mask_from_Quantity(mask_data)
    nside_in = hp.get_nside(mask_data)
    if nside_in == nside_out:
        logger.debug(f"Mask resolution matches map resolution. In: {nside_in}, Out: {nside_out}. Map not downgraded.")
        return mask_data
    elif nside_in < nside_out:
        logger.warning(f"Mask resolution is lower than map resolution. Consider scaling it externally. This is an unhandled case. Proceed with caution.")
    downgraded_mask = hp.ud_grade(mask_data, nside_out)
    mask = apply_threshold(downgraded_mask, threshold)
    return mask


def apply_threshold(mask, thresh):
    """
    Apply a threshold to a mask.

    Args:
        mask (np.ndarray): Numpy array representing the input mask data.
        thresh (float): The threshold to apply to the mask.

    Returns:
        np.ndarray: The mask after applying the threshold.
    """
    # Per Planck 2015 results:IX. Diffuse component separation: CMB maps
    #    When downscaling mask maps; threshold the downscaled map
    #    They use 0.9
    return np.where(mask<thresh, 0, 1)


def simple_galactic_mask(nside, width=10, smoothing=1):
    mask = np.ones(hp.nside2npix(nside))
    mask[hp.query_strip(nside, np.radians(90 - width/2), np.radians(90 + width /2))] = 0
    mask = hp.smoothing(mask, fwhm=np.radians(smoothing))
    return mask
