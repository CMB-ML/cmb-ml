from typing import List, Union, Tuple
from contextlib import contextmanager

import numpy as np
from astropy.units import Quantity


"""
I'm trying to figure out if there's a way to do these transformations as needed,
capturing them with a single internal object, tracking units, and able to transform
to whatever format is needed elsewhere.

Eventually, I guess.
"""


def convert_pysm3_to_hp(data: List[Quantity]) -> Tuple[List[np.ndarray], List[str]]:
    """
    PySM3 format is typically either a Quantity or list of Quantity objects.

    Healpy expects lists of np.ndarrays and associated units.
    """
    # Handle Quantity objects first
    if isinstance(data, list):
        if isinstance(data[0], Quantity):
            if column_units is None:
                column_units = [datum.unit for datum in data]
            data = [datum.value for datum in data]
    if isinstance(data, Quantity):
        column_units = [data.unit for _ in range(len(data))]
        data = data.value

    # Convert np.ndarrays of higher dimension to a list of 1D np.ndarrays
    if isinstance(data, np.ndarray) and data.shape[0] == 3:
        temp_data = []
        for i in range(3):
            temp_data.append(data[i, :])
        data = temp_data

    # For lists of np.ndarrays (most conditions from above), squeeze out extra dimensions
    if isinstance(data, list):
        data = [datum.squeeze() for datum in data]
    # For singular np.ndarrays (the remaining conditions), squeeze out extra dimensions
    elif isinstance(data, np.ndarray) and data.shape[0] == 1:
        data = data.squeeze()

    # Ensure that column units are strings, and not astropy Units, which don't have a good __str__
    if column_units:
        column_units = [str(unit) for unit in column_units]

    return (data, column_units)


def convert_to_ndarray():
    """
    PyTorch uses tensors, which are a stone's throw from higher-dimension
    np.ndarrays
    """


