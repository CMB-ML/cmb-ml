import numpy as np

import torch


class AbsMaxScaleMapAbstract(object):
    """
    Scales a map according to scale factors determined in a previous stage.
    Follows Petroff's method, despite being a less common approach.

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
        obs, cmb = map_data
        obs_scale = obs / self.obs_abs_max * 0.5 + 0.5
        cmb_scale = cmb / self.cmb_abs_max * 0.5 + 0.5
        return obs_scale, cmb_scale


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
        obs = map_data
        obs_scale = obs / self.obs_abs_max * 0.5 + 0.5
        return obs_scale


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
        obs, cmb = map_data
        obs_scale = (obs - 0.5) * 2 * self.obs_abs_max
        cmb_scale = (cmb - 0.5) * 2 * self.cmb_abs_max
        return obs_scale, cmb_scale


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
        obs = map_data
        obs_scale = (obs - 0.5) * 2 * self.obs_abs_max
        return obs_scale



class MinMaxScaleMap(object):
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
        obs, cmb = map_data
        obs_norm = (obs - self.obs_min) / (self.obs_max - self.obs_min)
        cmb_norm = (cmb - self.cmb_min) / (self.cmb_max - self.cmb_min)
        return obs_norm, cmb_norm
