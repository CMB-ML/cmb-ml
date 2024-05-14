import logging
import healpy as hp


logger = logging.getLogger(__name__)


def downgrade_mask(mask_data, nside_out):
    nside_in = hp.get_map_size(mask_data)
    if nside_in == nside_out:
        logger.info(f"Mask resolution matches map resolution. In: {nside_in}, Out: {nside_out}. No action taken.")
        return mask_data
    elif nside_in < nside_out:
        logger.warning(f"Mask resolution is lower than map resolution. Consider scaling it externally. This is an unhandled case. Proceed with caution.")
    downgraded_mask = hp.ud_grade(mask_data, nside_out)
    conservative_mask = (downgraded_mask > 0.0).astype(int)
    return conservative_mask
