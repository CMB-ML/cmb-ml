from .physics_instrument_noise_empty import EmptyNoise
from .physics_instrument_noise_variance import VarianceNoise
from .physics_instrument_noise_spatial_corr import SpatialCorrNoise
from .physics_instrument_noise_corr_aniso_nonstat import CorrAnisoNoise


def get_noise_class(label):
    if label == 'empty':
        return EmptyNoise
    elif label == 'variance':
        return VarianceNoise
    elif label == 'spatial_corr':
        return SpatialCorrNoise
    elif label == 'corr_aniso_nonstat':
        return CorrAnisoNoise
    else:
        raise ValueError(f"Unsupported noise type: {label}")
