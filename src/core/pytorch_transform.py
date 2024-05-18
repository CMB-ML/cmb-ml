import torch


class ToTensor(object):
    """Convert ndarrays in sample to Tensors of predefined dtype."""
    def __init__(self, dtype, device):
        self.dtype = dtype
        self.device = device

    def __call__(self, data):
        raise NotImplementedError("Abstract class")

    def ensure_tensor(self, item, dtype):
        # If item is already a tensor, set dtype and device without copying
        if isinstance(item, torch.Tensor):
            return item.to(dtype=dtype, device=self.device)
        # If item is not a tensor, make it one
        else:
            return torch.tensor(item, dtype=dtype, device=self.device)


class TrainToTensor(ToTensor):
    """Convert ndarrays in sample to Tensors of predefined dtype."""
    def __call__(self, data):
        obs, cmb = data
        obs = self.ensure_tensor(obs, dtype=self.dtype)
        cmb = self.ensure_tensor(cmb, dtype=self.dtype)
        return obs, cmb


class TestToTensor(ToTensor):
    """Convert ndarrays in sample to Tensors of predefined dtype."""
    def __call__(self, data):
        obs = data
        obs = self.ensure_tensor(obs, dtype=self.dtype)
        return obs
