import numpy as np


def prep_scaling(data: np.ndarray, vmin: np.ndarray, vmax: np.ndarray):
    if data.ndim == 1:
        return vmin, vmax
    reshape_dims = (data.shape[0],) + (1,) * (data.ndim - 1)
    vmin = vmin.reshape(reshape_dims)
    vmax = vmax.reshape(reshape_dims)
    return vmin, vmax


def minmax_scale(data: np.ndarray, vmin: np.ndarray, vmax: np.ndarray):
    vmin, vmax = prep_scaling(data, vmin, vmax)
    return (data - vmin) / (vmax - vmin)


def minmax_unscale(data: np.ndarray, vmin: np.ndarray, vmax: np.ndarray):
    vmin, vmax = prep_scaling(data, vmin, vmax)
    return data * (vmax - vmin) + vmin


class MinMaxScaler:
    def __init__(self, vmins: np.ndarray, vmaxs: np.ndarray):
        """
        Parameters
        ----------
        vmins : np.ndarray
            The minimum values for each field.
        vmaxs : np.ndarray
            The maximum values for each field.
        """
        self.vmin = vmins
        self.vmax = vmaxs

    def __call__(self, data):
        return minmax_scale(data, self.vmin, self.vmax)
