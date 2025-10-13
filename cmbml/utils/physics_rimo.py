# Produced with (significant cajoling of) ChatGPT 5.0
import numpy as np
import pysm3.units as u


def trim_by_mass(freq, tx, keep=0.999):
    """Keep the smallest contiguous frequency interval containing `keep` fraction of total area.
    Rescale so that the peak/height matches the original scale.
    """
    try:
        f_unit = freq.unit
    except:
        f_unit = None
    f = np.asarray(freq)
    t = np.clip(np.asarray(tx), 0, None)
    order = np.argsort(f)
    f = f[order]; t = t[order]

    # normalize for cumulative mass selection only
    w = t / t.sum()
    c = np.cumsum(w)

    # indices spanning desired mass fraction
    lo_idx = np.searchsorted(c, (1 - keep) / 2, side="left")
    hi_idx = np.searchsorted(c, 1 - (1 - keep) / 2, side="right")

    f_keep = f[lo_idx:hi_idx]
    t_keep = t[lo_idx:hi_idx]

    # renormalize so that total area under curve matches original
    scale = t.sum() / t_keep.sum()
    t_keep = t_keep * scale

    if f_unit is not None:
        f_keep = u.Quantity(f_keep, f_unit)
    return f_keep, t_keep


def get_eff_cen_freq(freq, tx):
    freq_eff = np.trapz(freq * tx, freq) / np.trapz(tx, freq)
    return freq_eff


def _cumulative_trapz(x, y):
    return np.concatenate(([0.0], np.cumsum(0.5*(y[1:]+y[:-1])*(x[1:]-x[:-1]))))

def compress_nonuniform(freq, tx, n=200, alpha=0.5):
    """
    Non-uniform conservative bandpass compression.
    Preserves total area and matches support: w_new[0]=w_new[-1]=0.

    Parameters
    ----------
    freq : array_like
        Frequencies (will be sorted).
    tx : array_like
        Transmission values (non-negative recommended).
    n : int
        Number of *interior* bins (excluding the zero endpoints).
    alpha : float
        Controls how non-uniform the sampling is:
        alpha=0 -> uniform frequency spacing,
        alpha=1 -> density prop_to w(f), i.e. denser where transmission is large.

    Returns
    -------
    f_new : ndarray
        New (non-uniform) frequency samples, including endpoints.
    t_new : ndarray
        Compressed transmission values, 0 at endpoints.
    """
    try:
        f_unit = freq.unit
    except:
        f_unit = None

    # sort inputs
    idx = np.argsort(freq)
    f = np.asarray(freq, float)[idx]
    y = np.asarray(tx, float)[idx]

    # define sampling density function
    s = np.maximum(y, 0.0)**alpha + 1e-300
    S = _cumulative_trapz(f, s)
    Stot = S[-1]

    # true throughput CDF for area preservation
    F = _cumulative_trapz(f, y)

    # choose edges: equal steps in cumulative s(f)
    # (n interior bins, +2 for endpoints)
    S_targets = np.linspace(0, Stot, n + 1)
    edges_inner = np.interp(S_targets, S, f)

    # ensure edges include exact endpoints
    f0, fN = f[0], f[-1]
    if edges_inner[0] > f0 or edges_inner[-1] < fN:
        edges = np.concatenate(([f0], edges_inner, [fN]))
    else:
        edges = edges_inner.copy()
        edges[0], edges[-1] = f0, fN

    # compute per-bin integrated area and average height
    F_edges = np.interp(edges, f, F)
    bin_area = np.diff(F_edges)
    widths = np.diff(edges)
    y_bin = bin_area / widths

    # bin centers
    f_bin = 0.5 * (edges[:-1] + edges[1:])

    # mass centroids
    fw = f * y
    M = _cumulative_trapz(f, fw)
    M_edges = np.interp(edges, f, M)
    with np.errstate(invalid='ignore', divide='ignore'):
        x_centroid = (M_edges[1:] - M_edges[:-1]) / bin_area
    mask = np.isfinite(x_centroid)
    f_bin[mask] = x_centroid[mask]

    # prepend/append zeros for anchored support
    f_new = np.concatenate(([f0], f_bin, [fN]))
    t_new = np.concatenate(([0.0], y_bin, [0.0]))

    # renormalize area
    A_orig = np.trapz(y, f)
    A_new = np.trapz(t_new, f_new)
    if A_new > 0:
        t_new *= (A_orig / A_new)

    if f_unit is not None:
        f_new = u.Quantity(f_new, f_unit)
    return f_new, t_new


def reduce_rimo(freq, tx, trim_to_mass=0.999, compress_to_n=50, compress_alpha=0.5):
    f, t = trim_by_mass(freq, tx, trim_to_mass)
    f, t = compress_nonuniform(f, t, compress_to_n, compress_alpha)
    return f, t
