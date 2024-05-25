from typing import List, Dict, Union
import logging
from pathlib import Path

import numpy as np

from src.core import GenericHandler, register_handler
from src.core import make_directories


logger = logging.getLogger(__name__)


class NumpyMap(GenericHandler):
    def read(self, path: Union[Path, str]):
        data = np.load(path)
        return data

    def write(self, 
              path: Union[Path, str], 
              data: Union[List[np.ndarray], np.ndarray]) -> None:
        # TODO: Better docstring
        """
        Writes map data to a numpy file.
        Converts lists of arrays to a single array.
        Does not handle units or fields information; this is retained for interchangeability with Healpy
        """
        # TODO: handle either list of np.ndarrays or a single np.ndarray in other AssetHandlers as well.
        path = Path(path)
        make_directories(path)
        # parent = full_path.parent
        # filename = full_path.name

        data = handle_list_of_arrays(data)

        # Heads up: CMBNNCS parameter names are potentially confusing.
        # savenpy(path=parent,
        #         FileName=filename,
        #         File=data,
        #         dtype=data.dtype)

        np.save(path, data)


def handle_list_of_arrays(data: Union[List[np.ndarray], np.ndarray]) -> np.ndarray:
    """
    Processes an input that can be either a single numpy ndarray or a list of ndarrays.
    If the input is a list of ndarrays, this function stacks them into a single ndarray along a new axis.
    If the input is a single ndarray, it returns the ndarray unchanged.

    Parameters:
    -----------
    data : np.ndarray or list of np.ndarray
        The input data to process. It can be a single numpy ndarray or a list containing only numpy ndarrays.

    Returns:
    --------
    np.ndarray
        If `data` is a list of ndarrays, returns a new ndarray which is the result of stacking the input ndarrays along a new axis.
        If `data` is a single ndarray, returns the ndarray itself.

    Raises:
    -------
    ValueError
        If `data` is neither an ndarray nor a list of ndarrays, or if any element within a list is not an ndarray.

    Examples:
    ---------
    >>> arr1 = np.array([1, 2, 3])
    >>> arr2 = np.array([4, 5, 6])
    >>> arr3 = np.array([7, 8, 9])
    >>> handle_list_of_arrays([arr1, arr2, arr3])
    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])

    >>> handle_list_of_arrays(arr1)
    array([1, 2, 3])
    """
    if isinstance(data, list):
        if all(isinstance(arr, np.ndarray) for arr in data):
            if not all(arr.dtype == data[0].dtype for arr in data):
                raise ValueError("All numpy arrays must have the same dtype")
            return np.stack(data, axis=0)
        else:
            raise ValueError("All items in the list must be numpy arrays")
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise ValueError("Input must be a numpy array or a list of numpy arrays")


# class ManyNumpyMaps(GenericHandler):
#     def __init__(self, experiment: ExperimentParameters) -> None:
#         super().__init__(experiment)
#         self.handler: NumpyMap = NumpyMap(experiment)

#     def read(self, path: Union[Path, str]) -> Dict[str, np.ndarray]:
#         path = Path(path)
#         maps = {}
#         for det in self.experiment.detector_freqs:
#             fn_template = path.name
#             fn = fn_template.format(det=det)
#             this_path = path.parent / fn
#             maps[det] = self.handler.read(this_path)
#         return maps

#     def write(self, path: Union[Path, str], data: Dict[str, np.ndarray], units=None, fields=None) -> None:
#         path = Path(path)
#         for det, map_to_write in data.items():
#             if fields is not None and len(fields) > 1:
#                 if len(fields) > 1:
#                     use_fields = fields
#                     # TODO: handle using intstrument knowledge
#                     if det in [545, 857]:
#                         use_fields = [fields[0]]
#             else:
#                 use_fields = None
#             fn_template = path.name
#             fn = fn_template.format(det=det)
#             this_path = path.parent / fn
#             self.handler.write(this_path, map_to_write, units=units, fields=use_fields)


register_handler("NumpyMap", NumpyMap)
# register_handler("ManyNumpyMaps", ManyNumpyMaps)
