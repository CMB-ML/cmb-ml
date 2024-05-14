from typing import Any, Dict, List, Union
import shutil
from pathlib import Path
import yaml
import logging

import numpy as np
import healpy as hp
from astropy.units import Quantity

from .asset_handler_registration import register_handler

logger = logging.getLogger(__name__)


class GenericHandler:
    # def __init__(self) -> None:
    #     pass

    def read(self, path: Path):
        raise NotImplementedError("This read() should be implemented by children classes.")

    def write(self, path: Path, data: Any):
        raise NotImplementedError("This write() should be implemented by children classes.")


class EmptyHandler(GenericHandler):
    def read(self, path: Path):
        raise NotImplementedError("This is a no-operation placeholder and has no read() function.")

    def write(self, path: Path, data: Any=None):
        make_directories(path)
        if data:
            raise NotImplementedError("This is a no-operation placeholder and has no write() function.")


class HealpyMap(GenericHandler):
    def read(self, path: Union[Path, str], map_fields=None, precision=None):
        path = Path(path)
        try:
            this_map: np.ndarray = hp.read_map(path, field=map_fields)
        except IndexError as e:
            if isinstance(map_fields, int):
                raise e
            elif len(map_fields) > 1:
                map_fields = tuple([0])
                this_map = hp.read_map(path, field=map_fields)
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

class Config(GenericHandler):
    def read(self, path: Path) -> Dict:
        logger.debug(f"Reading config from '{path}'")
        with open(path, 'r') as infile:
            data = yaml.safe_load(infile)
        return data

    def write(self, path, data) -> None:
        logger.debug(f"Writing config to '{path}'")
        make_directories(path)
        unnumpy_data = _convert_numpy(data)
        with open(path, 'w') as outfile:
            yaml.dump(unnumpy_data, outfile, default_flow_style=False)


class Mover(GenericHandler):
    def read(self, path: Path) -> None:
        raise NotImplementedError("No read method implemented for Mover Handler; implement a handler for files to be read.")

    def write(self, path: Path, source_location: Union[Path, str]) -> None:
        make_directories(path)
        # Move the file from the temporary location
        destination_path = Path(path).parent / str(source_location)
        logger.debug(f"Moving from {source_location} to {destination_path}")
        try:
            # Duck typing for more meaningful error messages
            source_path = Path(source_location)
        except Exception as e:
            # TODO: Better except here
            raise e
        shutil.copy(source_path, destination_path)
        source_path.unlink()


def _convert_numpy(obj: Union[Dict[str, Any], List[Any], np.generic]) -> Any:
    # Recursive function
    # The `Any`s in the above signature should be the same as the signature itself
    # GPT4, minimal modification
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: _convert_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy(item) for item in obj]
    else:
        return obj


def make_directories(path: Union[Path, str]) -> None:
    path = Path(path)
    folders = path.parent
    folders.mkdir(exist_ok=True, parents=True)


register_handler("EmptyHandler", EmptyHandler)
register_handler("HealpyMap", HealpyMap)
register_handler("Config", Config)
register_handler("Mover", Mover)
