import numpy as np

def get_np_dtype(cfg_str):
    if cfg_str == "float":
        this_dtype = np.float32
    elif cfg_str == "double":
        this_dtype = np.float64
    else:
        raise ValueError('Only "float" and "double" are accepted as dtypes.')
    return this_dtype
