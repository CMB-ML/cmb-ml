import numpy as np

import torch


def min_max_scale(data, min_v, max_v):
    return (data - min_v) / (max_v - min_v)


def min_max_unscale(data, min_v, max_v):
    return data * (max_v - min_v) + min_v


class MinMaxScaleMapAbstract(object):
    """
    Scales a map according to scale factors determined in a previous stage.
    More typical method is min-max scaling.

    Args:
        all_map_fields (str): The configuration file path.
        scale_factors (str): The stage string. (To be removed?)
        detector_fields (str): The name of the split.
    """
    def __init__(self, 
                 all_map_fields: str, 
                 scale_factors: dict, 
                 dtype,
                 device: str="cpu"):
        cmb = scale_factors.pop("cmb")

        freqs = scale_factors.keys()
        n_freqs = len(freqs)
        n_map_fields = len(all_map_fields)

        # If values are not set for any pair of min, max; we assume it's at a freq and field 
        #    that will be zero padded.
        obs_min = np.zeros(shape=(n_freqs, n_map_fields))
        obs_max = np.ones(shape=(n_freqs, n_map_fields))
        # Iterate through ordered frequencies and fields to align min and max values
        for i, freq in enumerate(freqs):
            for j, field in enumerate(all_map_fields):
                if field in scale_factors[freq]:
                    obs_min[i, j] = scale_factors[freq][field]['min_val']
                    obs_max[i, j] = scale_factors[freq][field]['max_val']

        cmb_min = np.zeros(shape=(1, n_map_fields))
        cmb_max = np.ones(shape=(1, n_map_fields))
        for j, field in enumerate(all_map_fields):
            cmb_min[0, j] = cmb[field]['min_val']
            cmb_max[0, j] = cmb[field]['max_val']

        # Convert lists to tensors
        self.obs_min = torch.tensor(obs_min, dtype=dtype, device=device)
        self.obs_max = torch.tensor(obs_max, dtype=dtype, device=device)

        self.cmb_min = torch.tensor(cmb_min, dtype=dtype, device=device)
        self.cmb_max = torch.tensor(cmb_max, dtype=dtype, device=device)

    def __call__(self, map_data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Abstract class. Use a Train or Test version.")


class TrainMinMaxScaleMap(MinMaxScaleMapAbstract):
    """
    Scales a map according to scale factors determined in a previous stage.
    Follows Petroff's method, despite being a less common approach.

    Args:
        all_map_fields (str): The configuration file path.
        scale_factors (str): The stage string. (To be removed?)
        detector_fields (str): The name of the split.
    """
    def __call__(self, map_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            map_data (tuple(torch.tensor)): 
                A tuple containing two tensors:
                obs (batch x N_dets x N_pix tensor): observation maps
                cmb (batch x 1 x N_pix tensor): cmb map
        """
        obs, cmb = map_data
        return min_max_scale(obs, self.obs_min, self.obs_max), min_max_scale(cmb, self.cmb_min, self.cmb_max)


class TestMinMaxScaleMap(MinMaxScaleMapAbstract):
    """
    Scales a map according to scale factors determined in a previous stage.
    Follows Petroff's method, despite being a less common approach.

    Args:
        all_map_fields (str): The configuration file path.
        scale_factors (str): The stage string. (To be removed?)
        detector_fields (str): The name of the split.
    """
    def __call__(self, map_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            map_data (torch.tensor): 
                obs (batch x N_dets x N_pix tensor): observation maps
        """
        obs = map_data
        return min_max_scale(obs, self.obs_min, self.obs_max)


class TrainMinMaxUnScaleMap(MinMaxScaleMapAbstract):
    """
    UnScales a map according to scale factors determined in a previous stage.
    Follows Petroff's method, despite being a less common approach.

    Args:
        all_map_fields (str): The configuration file path.
        scale_factors (str): The stage string. (To be removed?)
        detector_fields (str): The name of the split.
    """
    def __call__(self, map_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            map_data (tuple(torch.tensor)): 
                A tuple containing two tensors:
                obs (batch x N_dets x N_pix tensor): observation maps
                cmb (batch x 1 x N_pix tensor): cmb map
        """
        obs, cmb = map_data
        return min_max_unscale(obs, self.obs_min, self.obs_max), min_max_unscale(cmb, self.cmb_min, self.cmb_max)


class TestMinMaxUnScaleMap(MinMaxScaleMapAbstract):
    """
    UnScales a map according to scale factors determined in a previous stage.
    Follows Petroff's method, despite being a less common approach.

    Args:
        all_map_fields (str): The configuration file path.
        scale_factors (str): The stage string. (To be removed?)
        detector_fields (str): The name of the split.
    """
    def __call__(self, map_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            map_data (tensor): 
                cmb (batch x 1 x N_pix tensor): cmb map
        """
        cmb = map_data
        return min_max_unscale(cmb, self.cmb_min, self.cmb_max)
