import numpy as np
import healpy as hp

from cmbml.utils.physics_mask import downgrade_mask


def get_inverse_nside(nside_obs, nside_patches):
    nside_ip = nside_obs // nside_patches
    return nside_ip


# Function to determine which top level pixels are valid
def get_valid_ids(mask, nside_obs, nside_patches, threshold=0.9):
    nside_ip = get_inverse_nside(nside_obs, nside_patches)
    # mask_ip = downgrade_mask(mask, nside_ip)
    mask_ip = downgrade_mask(mask.astype(float), nside_ip, threshold=threshold)
    valid_ids = np.where(mask_ip==1)[1]
    return valid_ids


# Functions to get the pixels in a patch (to end of file)
def get_nest_order_within_patch(nside_patch):
    n_bits = int(np.log2(nside_patch))

    binary_vals = np.array([list(np.binary_repr(i, width=n_bits)) for i in range(nside_patch)], dtype=int)

    row_bits = binary_vals[:, None, :]
    col_bits = binary_vals[None, :, :]

    interleaved_bits = np.zeros((nside_patch, nside_patch, n_bits*2), dtype=int)
    interleaved_bits[:, :, 0::2] = row_bits
    interleaved_bits[:, :, 1::2] = col_bits

    out = np.sum(interleaved_bits * (2 ** np.arange(n_bits * 2)[::-1]), axis=-1)
    return out


def get_patch_pixels(ring_patch_id: int, nside_patch: int, nside_obs: int):
    """
    Get the pixels in a patch

    Args:
        ring_patch_id: The ring index of the patch (as though the patch were a 
                       single HEALPix pixel)
        nside_patch: The nside of the patch
        nside_obs: The nside of the observation

    Returns:
        r_is: The ring indices of the pixels in the patch
    """
    nside_ip = get_inverse_nside(nside_obs, nside_patch)
    n_pid = hp.ring2nest(nside_ip, ring_patch_id)
    npix_tgt = hp.nside2npix(nside_obs)
    npix_pch = hp.nside2npix(nside_ip)

    nest_idx_i = n_pid * npix_tgt // npix_pch
    nest_idx_f = (n_pid + 1) * npix_tgt // npix_pch

    r_is = []
    for n_i in range(nest_idx_i, nest_idx_f):
        r_i = hp.nest2ring(nside_obs, n_i)
        r_is.append(r_i)
    r_is = np.array(r_is)

    nest_order_within_patch = get_nest_order_within_patch(nside_obs // nside_ip)
    r_is = r_is[nest_order_within_patch]

    return r_is


def make_pixel_index_lut(nside_obs, nside_patches):
    nside_ip = get_inverse_nside(nside_obs, nside_patches)
    n_patches = hp.nside2npix(nside_ip)
    lut = []
    for i in range(n_patches):
        lut.append(get_patch_pixels(i, nside_patches, nside_obs))
    lut = np.array(lut)
    return lut
