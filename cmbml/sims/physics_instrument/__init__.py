from .physics_instrument_noise_empty import EmptyNoise
from .physics_instrument_noise_variance import VarianceNoise


def get_noise_class(label):
    if label == 'empty':
        return EmptyNoise
    elif label == 'variance':
        return VarianceNoise
    else:
        raise ValueError(f"Unsupported noise type: {label}")
