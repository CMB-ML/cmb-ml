import numpy as np

import torch

from .scale_find_absmax import abs_max_scale, abs_max_unscale


class AbsMaxScaleMapAbstract(object):
    """
    Scales a map according to scale factors determined in a previous stage.
    Follows Petroff's method, despite being a less common approach.

    Args:
        all_map_fields (str): The configuration file path.
        scale_factors (str): The stage string. (To be removed?)
        dtype (torch.dtype): The name of the split.
        device (str): The device to use, following PyTorch conventions.
    """
    def __init__(self, 
                 all_map_fields: str, 
                 scale_factors: dict, 
                 dtype: torch.dtype,
                 device: str="cpu"):
        cmb_sub_dict = scale_factors["cmb"]

        freqs = [k for k in scale_factors.keys() if k != "cmb"]
        n_freqs = len(freqs)
        n_map_fields = len(all_map_fields)

        # If values are not set for any pair of min, max; we assume it's at a freq and field 
        #    that will be zero padded.
        obs_abs_max = np.ones(shape=(n_freqs, n_map_fields))
        # Iterate through ordered frequencies and fields to align min and max values
        for i, freq in enumerate(freqs):
            for j, field in enumerate(all_map_fields):
                if field in scale_factors[freq]:
                    obs_abs_max[i, j] = scale_factors[freq][field]['abs_max']

        cmb_abs_max = np.ones(shape=(1, n_map_fields))
        for j, field in enumerate(all_map_fields):
            cmb_abs_max[0, j] = cmb_sub_dict[field]['abs_max']

        # Convert lists to tensors
        self.obs_abs_max = torch.tensor(obs_abs_max, dtype=dtype, device=device)

        self.cmb_abs_max = torch.tensor(cmb_abs_max, dtype=dtype, device=device)

    def __call__(self, map_data: torch.Tensor) -> torch.Tensor:
        """
        Abstract. To be implemented in base classes.

        Args:
            map_data (either tuple(torch.tensor) or torch.tensor): 
                The data to be pre- or post- processed.
        """
        raise NotImplementedError("Abstract class. Use TrainAbsMaxScaleMap or TestAbsMaxScaleMap")


class TrainAbsMaxScaleMap(AbsMaxScaleMapAbstract):
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
        return abs_max_scale(obs, self.obs_abs_max), abs_max_scale(cmb, self.cmb_abs_max)


class TestAbsMaxScaleMap(AbsMaxScaleMapAbstract):
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
        return abs_max_scale(obs, self.obs_abs_max)


class TrainAbsMaxUnScaleMap(AbsMaxScaleMapAbstract):
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
        return abs_max_unscale(obs, self.obs_abs_max), abs_max_unscale(cmb, self.cmb_abs_max)


class TestAbsMaxUnScaleMap(AbsMaxScaleMapAbstract):
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
        return abs_max_unscale(cmb, self.cmb_abs_max)
