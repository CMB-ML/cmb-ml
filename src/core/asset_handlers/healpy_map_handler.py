from typing import Any, Dict, List, Union
from pathlib import Path
import logging

import numpy as np
import healpy as hp
from astropy.units import Quantity

from core.asset_handlers import GenericHandler, make_directories
from .asset_handler_registration import register_handler


logger = logging.getLogger(__name__)


class HealpyMap(GenericHandler):
    def read(self, path: Union[Path, str], 
             map_fields=None, 
             precision=None, 
             read_to_nest:bool=None):
        path = Path(path)
        if read_to_nest is None:
            read_to_nest = False
        try:
            this_map: np.ndarray = hp.read_map(path, field=map_fields, nest=read_to_nest)
        except IndexError as e:
            # IndexError occurs if a map field does not exist for a given file - especially when trying to get polarization information from 545 or 857 GHz map
            if isinstance(map_fields, int):
                raise e
            elif len(map_fields) > 1:
                logger.warning("Defaulting to reading a single field from the file. The 857 and 545 maps have no polarization information. Consider suppressing this warning if running a large run.")
                map_fields = tuple([0])
                this_map = hp.read_map(path, field=map_fields, nest=read_to_nest)
            else:
                raise e
        except FileNotFoundError as e:
            raise FileNotFoundError(f"This map file cannot be found: {path}")
        # When healpy reads a single map that I've generated with simulations,
        #    the bite order is Big-Endian instead of native ('>' instead of '=')
        if this_map.dtype.byteorder == '>':
            # The byteswap() method swaps the byte order of the array elements
            # The newbyteorder() method changes the dtype to native byte order without changing the actual data
            this_map = this_map.byteswap().newbyteorder()
        # If a single field was requested, healpy.read_map with produce it as a 1D array
        #    for consistency, we want a 2D array
        if len(this_map.shape) == 1:
            this_map = this_map.reshape(1, -1)
        if precision == "float":
            this_map = this_map.astype(np.float32)
        return this_map

    def write(self, 
              path: Union[Path, str], 
              data: Union[List[Union[np.ndarray, Quantity]], np.ndarray],
              nest: bool = None,
              column_names: List[str] = None,
              column_units: List[str] = None,
              overwrite: bool = True
              ):
        # Format data as either a single np.ndarray or lists of np.ndarray without singular dimensions

        # Handle Quantity objects first
        if isinstance(data, list):
            if isinstance(data[0], Quantity):
                if column_units is None:
                    column_units = [datum.unit for datum in data]
                data = [datum.value for datum in data]
        if isinstance(data, Quantity):
            if column_units is None:
                column_units = data.unit
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

        path = Path(path)
        make_directories(path)
        hp.write_map(filename=path, 
                     m=data, 
                     nest=nest,
                     column_names=column_names,
                     column_units=column_units,
                     dtype=data[0].dtype, 
                     overwrite=overwrite)


register_handler("HealpyMap", HealpyMap)
