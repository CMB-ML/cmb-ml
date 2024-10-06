"""
Module Name: map_formats.py

This module contains a function for getting just the data from astropy Quantity objects (created by PySM3)
so that they can be used with Healpy. I do not recall which Healpy function required this.

Functions:
    convert_pysm3_to_hp: Convert PySM3 data to Healpy data.

Author: James Amato
Date: June 11, 2024
Version: 0.1.0

Edits: Sept 16, 2024 - Added documentation
"""
from typing import List, Union, Tuple
from contextlib import contextmanager

import numpy as np
from astropy.units import Quantity


def convert_pysm3_to_hp(data: List[Quantity]) -> Tuple[List[np.ndarray], List[str]]:
    """
    PySM3 format is typically either a Quantity or list of Quantity objects.

    Healpy expects lists of np.ndarrays and associated units.
    Args:
        data (List): The data in PySM3 format.

    Returns:
        Tuple: The data in Healpy format.
    """
    column_units = None
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


# def convert_to_ndarray():
#     """
#     PyTorch uses tensors, which are a stone's throw from higher-dimension
#     np.ndarrays
#     """
