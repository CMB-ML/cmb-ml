import numpy as np


def get_var_per_ell(ells, f_sky=1, N_ell=None):
    """
    Cosmic variance is the expected variation for a single
    $\ell$ over all $2\ell + 1$ modes.

    \[
        \frac{\Delta C_\ell}{C_\ell} = 
        \sqrt{
            \frac{2}{(2\ell + 1)f_\mathrm{sky}}
        }
    \]

    Assumes no noise spectrum.

    Parameters
    ells (np.ndarray): multipoles wanted
    f_sky (float): fraction of the sky
    N_ell (None): Noise per ell. Parameter is simply a reminder to the 
                  user that this makes the assumption of no noise.
    """
    if N_ell is not None:
        raise ValueError("Noise power spectrum is assumed to be zero." \
        " Another method is needed when accounting for noise.")
    if f_sky > 1 or f_sky <= 0:
        raise ValueError(f"Sky fraction (f_sky) must be in (0, 1]. Got {f_sky}")
    return np.sqrt(2 / ((2*ells + 1)*f_sky))
