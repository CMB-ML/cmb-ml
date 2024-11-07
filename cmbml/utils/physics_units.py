import pysm3.units as u
from cmbml.utils.fits_inspection import get_field_unit_str


def get_fields_units_from_fits(fits_fn, fields, hdu=1):
    """
    Get the units associated with specific fields from the header of a FITS file.

    Args:
        fits_fn (str): The filename of the FITS file.
        fields (List[int]): The indices of the fields.

    Returns:
        List[str]: The units of the fields.
    """

    units = []
    for field in fields:
        unit_str = get_field_unit_str(fits_fn, field, hdu=hdu)
        unit = convert_field_str_to_Unit(unit_str)
        units.append(unit)
    return units


def convert_field_str_to_Unit(unit_str):
    custom_units = {
            # 'uK_CMB': u.uK_CMB,
            'Kcmb': u.K_CMB,
            # 'K_CMB': u.K_CMB,
            'MJy/sr': u.MJy / u.sr,
            'Kcmb^2': u.K_CMB**2,
            '(K_CMB)^2': u.K_CMB**2,
            # 'K_CMB^2': u.K_CMB**2,
            # 'uK_CMB^2': u.uK_CMB**2,
            # '(uK_CMB)^2': u.uK_CMB**2,
            # '(MJy/sr)^2': (u.MJy / u.sr)**2,
            '(Mjy/sr)^2': (u.MJy / u.sr)**2,
            # 'MJy/sr^2': (u.MJy / u.sr)**2
        }
    if not isinstance(unit_str, str):
        try:
            unit_str = unit_str.item()
        except AttributeError:
            raise TypeError(f"Expected a string, but got {type(unit_str)}")

    try:
        # Attempt to parse with Astropy's Unit function
        return u.Unit(unit_str)
    except ValueError:  # Astropy throws a ValueError for unrecognized units
        if unit_str in custom_units.keys():
            return custom_units[unit_str]
        else:
            raise ValueError(f"Unit {unit_str} not recognized.")
