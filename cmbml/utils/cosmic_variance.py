import numpy as np


def get_var_per_ell(ells, f_sky=1, N_ell=None):
    """
    Compute the fractional cosmic variance for each multipole $\ell$.

    Cosmic variance arises because only $(2\ell + 1)$ modes are available
    on the sky at each multipole. The fractional $1\sigma$ uncertainty is

    .. math::

        \frac{\Delta C_\ell}{C_\ell} =
        \sqrt{\frac{2}{(2\ell + 1) f_{\mathrm{sky}}}}.

    This function assumes no noise spectrum.

    Parameters
    ----------
    ells : np.ndarray
        Multipoles ($\ell$ values) at which to evaluate the variance.
    f_sky : float, optional
        Observed sky fraction in (0, 1]. Default is 1 (full sky).
    N_ell : None
        Placeholder for a noise power spectrum. Must be None here,
        since this routine assumes no noise. A different method is
        needed if noise is included.

    Returns
    -------
    np.ndarray
        Fractional cosmic variance per $\ell$ (dimensionless).
        Multiply by $C_\ell$ or $D_\ell$ to obtain absolute error bars.

    Raises
    ------
    ValueError
        If `f_sky` is not in (0, 1] or if `N_ell` is not None.
    """
    if N_ell is not None:
        raise ValueError(
            "Noise power spectrum is assumed to be zero. "
            "Use a different method when accounting for noise."
        )
    if f_sky > 1 or f_sky <= 0:
        raise ValueError(f"Sky fraction (f_sky) must be in (0, 1]. Got {f_sky}")
    return np.sqrt(2 / ((2*ells + 1) * f_sky))
