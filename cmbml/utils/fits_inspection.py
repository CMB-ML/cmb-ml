"""
Module Name: fits_inspection.py

This module contains helper functions for examining and processing fits files.
One function, get_map_dtype, is from the PySM3 template.py file, with minimal alteration.

Functions:
    print_out_header: Print out the header of each HDU in a FITS file.
    get_num_all_fields_in_hdr: Get the number of columns in the specified HDU.
    get_num_field_types_in_hdr: Count the TTYPE# keywords in the header of the specified HDU.
    get_field_unit_str: Get the unit (TUNIT#) of a field in the specified HDU.
    get_other_info: Get the value of an arbitrary header keyword in the specified HDU.
    get_field_type_from_fits: Get the name (TTYPE#) of a field in the specified HDU.
    get_field_types_from_fits: Get the names of multiple (default: all) fields in the specified HDU.
    get_num_fields: Get the number of fields (TFIELDS) in each non-primary HDU.
    print_fits_information: Print out the information of a FITS file.
    get_fits_information: Get headers and field types/units for each non-primary HDU.
    show_all_maps: Display all maps in each HDU of a FITS file.
    show_one_map: Display a specific map from a FITS file.
    get_map_dtype: Get the data type of a map in a format compatible with numba and mpi4py.
    get_field_index_by_name: Find the index of a field in an HDU by its TTYPE name.
    find_field_across_hdus: Search all non-primary HDUs for a field and return (hdu, index) pairs.
    get_field_data: Load a single column from a FITS table HDU by name.
    require_field_index: Like get_field_index_by_name, but raises KeyError if the field is missing.

Author: 
Date: June 11, 2024
Version: 0.1.0

Edits: Sept 16, 2024 - Added documentation
        Sept 29, 2025 - Added functions for finding a particular field in a fits file
        Sept 18, 2026 - Corrected module name in docstring; updated function list to
                        match current names and added missing entries; corrected
                        get_other_info docstring; get_field_index_by_name now opens 
                        the file once; added require_field_index
        Sept 19, 2026 - print_out_header now shows comments and blank undefined values
"""
from typing import Dict, Union, Any

import numpy as np
import healpy as hp
from astropy.io import fits
import matplotlib.pyplot as plt
from pathlib import Path
from pprint import pprint


ASSUME_FITS_HEADER = 1



def _get_column_names(fits_fn, hdu=1) -> list[str]:
    with fits.open(fits_fn) as hdul:
        return list(hdul[hdu].columns.names)


def print_out_header(fits_fn):
    """
    Print out the header of each HDU in a FITS file.

    Args:
        fits_fn (str): The filename of the FITS file.
    """
    with fits.open(fits_fn) as hdul:
        for i, hdu in enumerate(hdul):
            print(f"Header for HDU {i}:")
            for card in hdu.header.cards:
                value = "" if isinstance(card.value, fits.card.Undefined) else card.value
                comment = f"  / {card.comment}" if card.comment else ""
                print(f"{card.keyword}: {value}{comment}")
            print("\n" + "-"*50 + "\n")


def get_num_all_fields_in_hdr(fits_fn, hdu) -> int:
    """
    Get the number of fields in the header of the specified HDU
    (Header Data Unit) in a FITS file.

    Args:
        fits_fn (str): The filename of the FITS file.
        hdu (int): The index of the HDU.

    Returns:
        The number of fields in the header.
    """

    with fits.open(fits_fn) as hdul:
        n_fields = len(hdul[hdu].columns)
    return n_fields


def get_num_field_types_in_hdr(fits_fn, hdu) -> int:
    """
    Get the number of fields in the header of the specified HDU
    (Header Data Unit) in a FITS file which are names TTYPE#.

    Args:
        fits_fn (str): The filename of the FITS file.
        hdu (int): The index of the HDU.

    Returns:
        The number of fields in the header.
    """
    with fits.open(fits_fn) as hdul:
        # iterate through column names and count the number of TTYPE# fields
        n_fields = 0
        for card in hdul[hdu].header.cards:
            if card.keyword.startswith("TTYPE"):
                n_fields += 1
    return n_fields


def get_field_unit_str(fits_fn, field_idx, hdu=1):
    """
    Get the unit associated with a specific field from the header of the 
    specified HDU (Header Data Unit) in a FITS file.

    Args:
        fits_fn (str): The filename of the FITS file.
        hdu (int): The index of the HDU.
        field_idx (int): The index of the field.

    Returns:
        str: The unit of the field.
    """
    with fits.open(fits_fn) as hdul:
        try:
            field_num = field_idx + 1
            unit = hdul[hdu].header[f"TUNIT{field_num}"]
        except KeyError:
            unit = ""
    return unit


def get_other_info(fits_fn, header_lbl, hdu=1):
    """
    Get the value of an arbitrary header keyword from the specified
    HDU (Header Data Unit) in a FITS file.

    Args:
        fits_fn (str): The filename of the FITS file.
        header_lbl (str): The header keyword to look up (e.g., "NSIDE", "ORDERING").
        hdu (int): The index of the HDU. Defaults to 1.

    Returns:
        The keyword's value, or "" if the keyword is not present.
    """
    with fits.open(fits_fn) as hdul:
        try:
            val = hdul[hdu].header[header_lbl]
        except KeyError:
            val = ""
    return val


def get_field_type_from_fits(fits_fn, field_idx, hdu):
    """
    Get the name of a specific field from the header of the specified HDU
    (Header Data Unit) in a FITS file.

    Args:
        fits_fn (str): The filename of the FITS file.
        hdu (int): The index of the HDU.
        field_idx (int): The index of the field.

    Returns:
        str: The name of the field.
    """
    with fits.open(fits_fn) as hdul:
        try:
            field_num = field_idx + 1
            name = hdul[hdu].header[f"TTYPE{field_num}"]
        except KeyError:
            name = ""
    return name


def get_field_types_from_fits(fits_fn, fields_idcs=None, hdu=1):
    """
    Get the names of all fields from the header of the specified HDU
    (Header Data Unit) in a FITS file.

    Args:
        fits_fn (str): The filename of the FITS file.
        hdu (int): The index of the HDU.

    Returns:
        List[str]: The names of the fields.
    """
    if fields_idcs is None:
        n_fields = get_num_field_types_in_hdr(fits_fn, hdu)
        fields_idcs = range(n_fields)
    names = []
    for field_idx in fields_idcs:
        name = get_field_type_from_fits(fits_fn, field_idx, hdu)
        names.append(name)
    return names


def get_num_fields(fits_fn) -> Dict[int, int]:
    """
    Get the number of fields in each HDU (Header Data Unit)
    of a FITS file.

    Args:
        fits_fn (str): The filename of the FITS file.

    Returns:
        A dictionary where the keys area the HDU indices and the
        values are the number of fields in each HDU.
    """
    # Open the FITS file
    n_fields = {}
    with fits.open(fits_fn) as hdul:
        # Loop over all HDUs in the FITS file
        for i, hdu in enumerate(hdul):
            if i == 0:
                continue  # skip 0; it's fits boilerplate
            for card in hdu.header.cards:
                if card.keyword == "TFIELDS":
                    n_fields[i] = card.value
    return n_fields


def print_fits_information(fits_fn) -> None:
    """
    Print out the information of a FITS file.

    Args:
        fits_fn (str): The filename of the FITS file.
    """
    fits_info = get_fits_information(fits_fn)
    pprint(fits_info)


def get_fits_information(fits_fn):
    """
    Get detailed information about a FITS file.

    Args:
        fits_fn (str): The filename of the FITS file.

    Returns:
        A nested dictionary where the keys of the top level are
        the indices of the HDUs (Header Data Units) and the values
        are dictionaries containing information about the HDU. These
        dictionaries contain the header of the HDU and a dictionary
        ("FIELDS") where the keys are the indices of each field and
        the values are dictionaries containing the type and unit of
        each field.
    """
    n_fields_per_hdu = get_num_fields(fits_fn)
    watch_keys = {}
    types_str_base = "TTYPE"
    units_str_base = "TUNIT"
    maps_info = {}
    for hdu_n in n_fields_per_hdu:
        indices = list(range(1, n_fields_per_hdu[hdu_n] + 1))
        field_types_keys = [f"{types_str_base}{i}" for i in indices]
        field_units_keys = [f"{units_str_base}{i}" for i in indices]
        watch_keys[hdu_n] = {"types": field_types_keys, "units": field_units_keys}

        maps_info[hdu_n] = {}
    with fits.open(fits_fn) as hdul:
        # Loop over all HDUs in the FITS file
        for hdu_n, hdu in enumerate(hdul):
            if hdu_n == 0:
                continue
            maps_info[hdu_n] = dict(hdu.header)
            maps_info[hdu_n]["FIELDS"] = {}
            # print(hdu_n, hdu.data.shape, '\n', hdu.data)
            for field_n in range(1, n_fields_per_hdu[hdu_n] + 1):
                field_info = {}
                # Construct the keys for type and unit
                ttype_key = f'TTYPE{field_n}'
                tunit_key = f'TUNIT{field_n}'

                # Retrieve type and unit for the current field, if they exist
                field_info['type'] = maps_info[hdu_n].get(ttype_key, None)
                field_info['unit'] = maps_info[hdu_n].get(tunit_key, None)

                # Add the field info to the fields_info dictionary
                maps_info[hdu_n]["FIELDS"][field_n] = field_info
    return maps_info


def show_all_maps(fits_fn):
    """
    Display all maps in each HDU (Header Data Unit)
    of a FITS file.

    Args:
        fits_fn (str): The filename of the FITS file.
    """
    n_fields_per_hdu = get_num_fields(fits_fn)
    for hdu_n, n_fields in n_fields_per_hdu.items():
        for field in range(n_fields):
            print(hdu_n, field)
            this_map = hp.read_map(fits_fn, hdu=hdu_n, field=field)
            hp.mollview(this_map)
            plt.show()


def show_one_map(fits_fn, hdu_n, field_n):
    """
    Display a specific map from a FITS file.

    Args:
        fits_fn (str): The filename of the FITS file.
        hdu_n (int): The number of the HDU (Header Data Unit).
        field_n (int): The number of the field.
    """
    this_map = hp.read_map(fits_fn, hdu=hdu_n, field=field_n)
    hp.mollview(this_map)
    plt.show()


def get_map_dtype(m: np.ndarray):
    """
    Get the data type of a map in a format compatible
    with numba and mpi4py.

    Args:
        m (np.ndarray): Numpy array representing the map.

    Returns:
        np.dtype: The data type of the map.
    """
    # From PySM3 template.py's read_map function, with minimal alteration:
    dtype = m.dtype
    # numba only supports little endian
    if dtype.byteorder == ">":
        dtype = dtype.newbyteorder()
    # mpi4py has issues if the dtype is a string like ">f4"
    if dtype == np.dtype(np.float32):
        dtype = np.dtype(np.float32)
    elif dtype == np.dtype(np.float64):
        dtype = np.dtype(np.float64)
    # End of used portion
    return dtype


def get_field_index_by_name(
    fits_fn: str,
    field_name: str,
    hdu: int = 1,
    case_sensitive: bool = False,
    match: str = "exact",  # "exact" | "prefix" | "contains"
) -> Union[int, None]:
    """
    Find the zero-based field index within a given HDU whose TTYPE matches `field_name`.

    Args:
        fits_fn (str): Path to the FITS file.
        field_name (str): Column/field name to look for (e.g., "II_COV").
        hdu (int, optional): HDU index to search (default: 1).
        case_sensitive (bool, optional): Whether to match case-sensitively (default: False).
        match (str, optional): Name matching mode:
            - "exact": names must match exactly
            - "prefix": column name must start with `field_name`
            - "contains": column name must contain `field_name` as a substring

    Returns:
        int | None: Zero-based field index if found, else None.
    """
    names = _get_column_names(fits_fn, hdu=hdu)

    def norm(s: str) -> str:
        return s.strip() if case_sensitive else s.strip().lower()

    target = norm(field_name)
    for i, nm in enumerate(norm(n or "") for n in names):
        if match == "exact" and nm == target:
            return i
        if match == "prefix" and nm.startswith(target):
            return i
        if match == "contains" and target in nm:
            return i
    return None


def find_field_across_hdus(
    fits_fn: str,
    field_name: str,
    case_sensitive: bool = False,
    match: str = "exact",
) -> list[tuple[int, int]]:
    """
    Search all non-primary HDUs for a field name and return the locations.

    Args:
        fits_fn (str): Path to the FITS file.
        field_name (str): Column/field name to look for.
        case_sensitive (bool, optional): Case sensitivity toggle.
        match (str, optional): "exact" | "prefix" | "contains"

    Returns:
        list[tuple[int, int]]: A list of (hdu_index, zero_based_field_index) matches.
    """
    results = []
    n_fields_per_hdu = get_num_fields(fits_fn)  # {hdu_idx: n_fields}
    for hdu in n_fields_per_hdu.keys():
        idx = get_field_index_by_name(
            fits_fn, field_name, hdu=hdu, case_sensitive=case_sensitive, match=match
        )
        if idx is not None:
            results.append((hdu, idx))
    return results


def get_field_data(
    fits_fn: Union[str, Path],
    field: str,
    hdu: Union[int, str] = 1,
    *,
    case_insensitive: bool = True,
    memmap: bool = True,
    squeeze: bool = True,
    copy: bool = False,
) -> Any:
    """
    Load a single column/field from a FITS table HDU.

    Parameters
    ----------
    fits_fn
        Path to the FITS file.
    field
        Column / field name in the table (e.g., 'TEMPERATURE', 'I', 'WEIGHT').
    hdu
        HDU index (int) or extension name (str). Default is 1 (common for tables).
    case_insensitive
        If True, match `field` ignoring case.
    memmap
        If True, allow memory-mapped reads (faster / lower RAM for big tables).
    squeeze
        If True, squeeze singleton dimensions (useful for scalar columns stored as (N,1)).
    copy
        If True, return a copy detached from the FITS memmap buffer.

    Returns
    -------
    Any
        Typically a NumPy array (for table columns). Could be a scalar type for single-row
        tables or a structured dtype element depending on FITS content.

    Raises
    ------
    FileNotFoundError
        If the FITS file does not exist.
    TypeError
        If the selected HDU is not a table HDU.
    KeyError
        If the requested field is not present.
    """
    fits_fn = Path(fits_fn)
    if not fits_fn.exists():
        raise FileNotFoundError(f"No such FITS file: {fits_fn}")

    with fits.open(fits_fn, memmap=memmap) as hdul:
        h = hdul[hdu]

        # Ensure it's a table HDU (BinTableHDU or TableHDU).
        if not isinstance(h, (fits.BinTableHDU, fits.TableHDU)):
            hdu_name = getattr(h, "name", "<unnamed>")
            raise TypeError(
                f"HDU {hdu!r} (name={hdu_name!r}) is type {type(h).__name__}, "
                "not a table HDU (BinTableHDU/TableHDU)."
            )

        data = h.data
        if data is None:
            raise ValueError(f"HDU {hdu!r} has no data.")

        # Determine available column names.
        colnames = list(data.columns.names) if hasattr(data, "columns") else list(data.names)

        # Find the requested column name.
        if case_insensitive:
            lookup = {c.lower(): c for c in colnames}
            key = lookup.get(field.lower())
        else:
            key = field if field in colnames else None

        if key is None:
            raise KeyError(
                f"Field {field!r} not found in HDU {hdu!r}. "
                f"Available fields: {colnames}"
            )

        arr = data[key]  # typically a numpy array view

        if squeeze:
            # Squeeze only if it's array-like
            try:
                arr = np.asarray(arr).squeeze()
            except Exception:
                pass

        if copy:
            try:
                arr = np.array(arr, copy=True)
            except Exception:
                # fallback for non-array returns
                pass

        return arr


def require_field_index(
    fits_fn: str,
    field_name: str,
    hdu: int = 1,
    case_sensitive: bool = False,
    match: str = "exact",
) -> int:
    """Like get_field_index_by_name, but raises KeyError if the field is missing."""
    idx = get_field_index_by_name(
        fits_fn, field_name, hdu=hdu, case_sensitive=case_sensitive, match=match
    )
    if idx is None:
        raise KeyError(
            f"Field {field_name!r} not found in HDU {hdu} of {fits_fn}. "
            f"Available fields: {_get_column_names(fits_fn, hdu)}"
        )
    return idx
