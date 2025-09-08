import importlib
import numpy as np
from skimage.metrics import (
                             peak_signal_noise_ratio,
                             mean_squared_error
                            #  structural_similarity
                             )


def get_func(func_str):
    """Dynamically imports a function from a given module path string."""
    try:
        if '.' in func_str:
            module_name, func_name = func_str.rsplit('.', 1)
            module = importlib.import_module(module_name)
            func = getattr(module, func_name)
        else:
            func = globals()[func_str]
    except KeyError:
        raise ImportError(f"No function named '{func_str}' found.")
    except ImportError as e:
        raise ImportError(f"Could not import module: {str(e)}")
    return func


def psnr(true, pred):
    data_range = get_data_range(true, pred)
    return peak_signal_noise_ratio(true, pred, data_range=data_range)


def rmse(true, pred):
    return np.sqrt(mean_squared_error(true, pred))


def ssim(true, pred):
    # Not currently implemented. From my brief literature review, it seems that SSIM tracks PSNR closely and was developed for human readability. I do not think it's appropriate here.
    data_range = get_data_range(true, pred)
    # return structural_similarity(true, pred, data_range=data_range)
    raise NotImplementedError("SSIM Requires arranging pixels as grids. Functions for such rearranging can come from the Healplay experiments, from Wang's method, or healpy's Cartesian projection (healpy.projector.CartesianProj).")


def get_data_range(true, pred):
    max_val = max(true.max(), pred.max())
    min_val = min(true.min(), pred.min())
    return max_val - min_val

