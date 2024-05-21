import torch


class ToTensor(object):
    """Convert ndarrays in sample to Tensors of predefined dtype."""
    def __init__(self, dtype=None, device=None):
        self.dtype = dtype
        self.device = device

    def __call__(self, data):
        raise NotImplementedError("Abstract class")

    def ensure_tensor(self, item):
        param_dict = dict()
        if self.dtype is not None:
            param_dict["dtype"] = self.dtype
        if self.device is not None:
            param_dict["device"] = self.device
        # If item is already a tensor, set dtype and device without copying
        if isinstance(item, torch.Tensor):
            return item.to(**param_dict)
        # If item is not a tensor, make it one
        else:
            return torch.tensor(item, **param_dict)


class TrainToTensor(ToTensor):
    """Convert ndarrays in sample to Tensors of predefined dtype."""
    def __call__(self, data):
        obs, cmb = data
        obs = self.ensure_tensor(obs)
        cmb = self.ensure_tensor(cmb)
        return obs, cmb


class TestToTensor(ToTensor):
    """Convert ndarrays in sample to Tensors of predefined dtype."""
    def __call__(self, data):
        obs = data
        obs = self.ensure_tensor(obs)
        return obs


def train_remove_map_fields(data):
    obs, cmb = data
    obs, cmb = obs.squeeze(1), cmb.squeeze(1)
    return obs, cmb

def test_remove_map_fields(data):
    cmb = data
    cmb = cmb.squeeze(1)
    return cmb
