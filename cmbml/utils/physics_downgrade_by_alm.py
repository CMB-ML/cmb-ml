import numpy as np
import healpy as hp
import pysm3.units as u


def downgrade_by_alm(some_map, target_nside):
    try:
        map_unit = some_map.unit
    except AttributeError:
        map_unit = None
    source_nside = hp.get_nside(some_map)
    assert target_nside <= source_nside/2, "Target nside must be less than the source nside"
    lmax_source = 3 * source_nside - 1
    alm = hp.map2alm(some_map, lmax=lmax_source)

    lmax_target = int(3 * target_nside - 1)
    alm_filter = np.zeros(lmax_source+1)
    alm_filter[:lmax_target+1] = 1
    alm_filtered = hp.almxfl(alm, alm_filter)
    some_map_filtered = hp.alm2map(alm_filtered, nside=target_nside)
    if map_unit is not None:
        some_map_filtered = u.Quantity(some_map_filtered, unit=map_unit)
    return some_map_filtered
