from .physics_instrument_noise_empty import EmptyNoise
from .physics_instrument_noise_variance import VarianceNoise
from .physics_instrument_noise_spatial_corr import SpatialCorrNoise


def get_noise_class(label):
    if label == 'empty':
        return EmptyNoise
    elif label == 'variance':
        return VarianceNoise
    elif label == 'spatial_corr':
        return SpatialCorrNoise
    else:
        raise ValueError(f"Unsupported noise type: {label}")
