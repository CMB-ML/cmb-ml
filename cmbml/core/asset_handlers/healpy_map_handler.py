from typing import Any, Dict, List, Union
from pathlib import Path
import logging

import numpy as np
import healpy as hp
import astropy.units as u
import astropy.io.fits as fits

from .asset_handlers_base import (
    GenericHandler, 
    register_handler, 
    make_directories)
from cmbml.utils.physics_units import get_fields_units_from_fits


logger = logging.getLogger(__name__)


def field_strs_to_ints(field_strs: Union[str, List[str]]) -> List[int]:
    if isinstance(field_strs, List):
        field_strs = "".join(field_strs)
    field_ints = []
    fields_lookup = {"I": 0, "Q": 1, "U": 2}
    for c in field_strs:
        field_ints.append(fields_lookup[c])
    return field_ints


class HealpyMap(GenericHandler):
    def read(self, 
             path: Union[Path, str],
             map_fields=None,
             map_field_strs=None, 
             precision=None, 
             read_to_nest:bool=None):

        if map_fields is None:
            if map_field_strs is None:
                map_fields = 0
            else:
                map_fields = field_strs_to_ints(map_field_strs)

        if precision is None:
            # Get and use the data type from the FITS file
            with fits.open(path) as hdul:
                use_field = map_fields if isinstance(map_fields, int) else map_fields[0]
                # HDUL[1] is the first extension of the FITS file; this is almost always the case
                #   And if it's not, you know it's such a weird situation that you'd check here
                dtype = hdul[1].columns[use_field].dtype
                # dtype is currently a weird numpy thing; get the actual class
                dtype = dtype.base.type
        else:
            dtype = precision

        path = Path(path)
        if read_to_nest is None:
            read_to_nest = False

        # Check this and crash early instead of after waiting to read the map
        map_units = get_fields_units_from_fits(path, map_fields)
        for map_unit in map_units:
            if map_unit != map_units[0]:
                raise ValueError("All fields in a map must have the same units.")
        map_unit = map_units[0]

        try:
            this_map: np.ndarray = hp.read_map(path, field=map_fields, nest=read_to_nest, dtype=dtype)
        except IndexError as e:
            # IndexError occurs if a map field does not exist for a given file - especially when trying to get polarization information from 545 or 857 GHz map
            logger.error(f"Map fields requested were {map_field_strs}, parsed as {map_fields}. Map at {path} does not have these fields. Note that field numbers for Healpy are 1 less than in the fits file.")
            raise e
            # if isinstance(map_field_strs, int):
            #     raise e
            # elif len(map_field_strs) > 1:
            #     logger.warning("Defaulting to reading a single field from the file. The 857 and 545 maps have no polarization information. Consider suppressing this warning if running a large run.")
            #     map_field_strs = tuple([0])
            #     this_map = hp.read_map(path, field=map_field_strs, nest=read_to_nest)
            # else:
        except FileNotFoundError as e:
            raise FileNotFoundError(f"This map file cannot be found: {path}")

        # If a single field was requested, healpy.read_map with produce it as a 1D array
        #    for consistency, we want a 2D array
        if len(this_map.shape) == 1:
            this_map = this_map.reshape(1, -1)
        # If the map is not a unitless map, convert it to the correct unit
        #   We check because otherwise the conversion will produce a 64 bit map regardless of the original type
        if not map_unit.is_equivalent(u.dimensionless_unscaled):
            this_map = u.Quantity(this_map, unit=map_unit, copy=False)
        return this_map

    def write(self, 
              path: Union[Path, str], 
              data: Union[List[Union[np.ndarray, u.Quantity]], np.ndarray],
              nest: bool = None,
              column_names: List[str] = None,
              column_units: List[str] = None,
              extra_header: List[str] = [],
              overwrite: bool = True
              ):
        """
        Writes a map to a file using healpy.write_map.

        Parameters:
        path (Union[Path, str]): The path to the file to write.
        data (Union[List[Union[np.ndarray, Quantity]], np.ndarray]): The map data to write.
        nest (bool): Whether to write the map in nested format.
        column_names (List[str]): The names of the columns in the map.
        column_units (List[str]): The units of the columns in the map.
        extra_header (List[str]): Extra header information to write.
                                  Format is [(<key>, <value>, <comment>), ...].
        overwrite (bool): Whether to overwrite the file if it already exists.
        """
        # Format data as either a single np.ndarray or lists of np.ndarray without singular dimensions

        # Handle Quantity objects first
        if isinstance(data, list):
            if isinstance(data[0], u.Quantity):
                if column_units is None:
                    column_units = [datum.unit.to_string().replace(' ', '') for datum in data]
                data = [datum.value for datum in data]
        if isinstance(data, u.Quantity):
            if column_units is None:
                logger.debug(f"Data is Quantity object with shape {data.shape}, no unit parameter provided to write(), setting units to array of {data.unit}, length {data.shape[0]}")
                if len(data.shape) == 1:  # One map in a shape (Npix, ) array
                    column_units = [data.unit]
                else:  # Maps in a shape (Nmaps, Npix) array
                    column_units = [data.unit]*data.shape[0]
            data = data.value

        # Convert np.ndarrays of higher dimension to a list of 1D np.ndarrays (we really should use hdf5 instead...)
        if isinstance(data, np.ndarray) and data.shape[0] == 3:
            temp_data = []
            for i in range(3):
                temp_data.append(data[i, :])
            data = temp_data

        # For lists of np.ndarrays (most conditions from above), squeeze out extra dimensions
        if isinstance(data, list):
            data = [datum.squeeze() for datum in data]
        # For singular np.ndarrays (the remaining conditions), squeeze out extra dimensions
        if isinstance(data, np.ndarray) and data.shape[0] == 1:
            data = data.squeeze()

        # Ensure that column units are strings
        if column_units:
            column_units = [str(unit) for unit in column_units]

        if extra_header is None:
            extra_header = []

        path = Path(path)
        make_directories(path)
        hp.write_map(filename=path, 
                     m=data, 
                     nest=nest,
                     column_names=column_names,
                     column_units=column_units,
                     extra_header=extra_header,
                     dtype=data[0].dtype,
                     overwrite=overwrite)


register_handler("HealpyMap", HealpyMap)
