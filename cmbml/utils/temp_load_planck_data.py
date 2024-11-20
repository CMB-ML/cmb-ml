import pysm3.units as u
import healpy as hp

from .get_planck_data import get_planck_obs_data, get_planck_pred_data


CENTER_FREQS = {
    30: 28.4 * u.GHz,      # Value from Planck DeltaBandpassTable & Planck 2018 I, Table 4
    44: 44.1 * u.GHz,      # Value from Planck DeltaBandpassTable & Planck 2018 I, Table 4
    70: 70.4 * u.GHz,      # Value from Planck DeltaBandpassTable & Planck 2018 I, Table 4
    100: 100.89 * u.GHz,   # Value from Planck DeltaBandpassTable
    143: 142.876 * u.GHz,  # Value from Planck DeltaBandpassTable
    217: 221.156 * u.GHz,  # Value from Planck DeltaBandpassTable
    353: 357.5 * u.GHz,    # Value from Planck DeltaBandpassTable
    545: 555.2 * u.GHz,    # Value from Planck DeltaBandpassTable
    857: 866.8 * u.GHz,    # Value from Planck DeltaBandpassTable
}


def load_planck_data_general(fn, detector):
    m = hp.read_map(fn)
    if detector in [545, 857]:
        map_unit = u.MJy / u.sr
    else:
        map_unit = u.K_CMB
    m = m * map_unit
    # print(f"{m.mean()}")
    m = m.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(CENTER_FREQS[detector]))
    # print(f"{m.mean()}")
    return m


def load_planck_obs_data(detector, assets_directory):
    fn = get_planck_obs_data(detector, assets_directory)
    return load_planck_data_general(fn, detector)

def load_planck_pred_data(detector, assets_directory):
    fn = get_planck_pred_data(detector, assets_directory)
    return load_planck_data_general(fn, detector)