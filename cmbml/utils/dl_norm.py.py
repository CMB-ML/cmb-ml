import numpy as np


def get_cl2dl_norm_per_ell(ells: np.ndarray) -> np.ndarray:
    """
    Compute the standard normalization factor that converts
    angular power spectra from $C_\ell$ to $D_\ell$.

    The relationship is

    .. math::

        D_\ell = \frac{\ell(\ell + 1)}{2\pi} \, C_\ell,

    where $C_\ell$ are the raw angular power spectrum coefficients
    (dimensionless) and $D_\ell$ provides a rescaled form commonly
    used in cosmology to emphasize features across multipoles.

    Parameters
    ----------
    ells : np.ndarray
        Array of multipole indices ($\ell$ values).

    Returns
    -------
    np.ndarray
        Normalization factor $\\ell(\\ell+1)/(2\\pi)$ for each input $\ell$.

    Notes
    -----
    - This function is agnostic to the specific spectrum type (TT, EE, BB, etc.),
      as the conversion from $C_\ell$ to $D_\ell$ is universal.
    - The returned normalization can be multiplied elementwise with a $C_\ell$
      array to yield the corresponding $D_\ell$.
    """
    return ells * (ells + 1) / (2 * np.pi)
