from typing import Any, Dict, List, Union
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

class NoHandler(GenericHandler):
    def read(self, path: Path):
        raise NotImplementedError("This is a no-operation placeholder and has no read() function.")

    def write(self, path: Path, data: Any):
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
        if map_fields is None or map_fields is int or len(map_fields) == 1:
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
        # Format data as np.ndarray without singular dimensions
        if isinstance(data, list):
            if isinstance(data[0], Quantity):
                data = [datum.value for datum in data]
            data = np.stack(data, axis=0)
        data = np.squeeze(data)
        # Get column units as a list of strings
        if column_units:
            column_units = [str(unit) for unit in column_units]
        elif isinstance(data, list):
            if isinstance(data[0], Quantity):
                column_units = [datum.unit.to_string() for datum in data]
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


register_handler("NoHandler", NoHandler)
register_handler("HealpyMap", HealpyMap)
# register_handler("ManyHealpyMaps", ManyHealpyMaps)
register_handler("Config", Config)

# TODO: Clarify about these
# register_handler("HealpyMapTemp", HealpyMapTemp)
# register_handler("ManyHealpyMapsTemp", ManyHealpyMapsTemp)