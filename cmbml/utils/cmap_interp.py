import numpy as np
from scipy.interpolate import interp1d

from matplotlib.colors import SymLogNorm, Normalize, ListedColormap


def get_cmap_interp(colors, num_points=1000):
    """
    Interpolate a colormap using linear interpolation for each channel (R, G, B).
    
    Parameters
    ----------
    colors : array-like
        Array of RGB values for the colormap. Expected shape is (N, 3).
    num_points : int, optional
        Number of points to interpolate between the original colors. The default is 1000.

    Returns
    -------
    cmap_interp : ListedColormap
        Interpolated colormap.
    """
    R = colors[:, 0]
    G = colors[:, 1]
    B = colors[:, 2]

    # Create an array for the positions of the original colors
    x = np.linspace(0, 1, len(R))  # Normalized x for interpolation

    # Step 2: Use linear interpolation for each channel
    linear_R = interp1d(x, R, kind='linear')
    linear_G = interp1d(x, G, kind='linear')
    linear_B = interp1d(x, B, kind='linear')

    # Interpolation over the fine range
    x_fine = np.linspace(0, 1, num_points)

    # Get interpolated values for each channel (linear interpolation between points)
    R_interp = linear_R(x_fine)
    G_interp = linear_G(x_fine)
    B_interp = linear_B(x_fine)

    # Combine the interpolated channels into a new colormap
    cmap_interp = np.transpose([R_interp, G_interp, B_interp])

    return cmap_interp.tolist()


def get_symlog_cmap(o_cmap, norm, total_points=2000):
    """
    From a source symmetric colormap, produce a new colormap to be used with 
    a SymLogNorm object, which preserves the neutral color at the center of the
    colormap.

    Parameters
    ----------
    o_cmap : ListedColormap
        Original colormap to be interpolated. Expects a ListedColormap object.
        I do not know if it will work with other colormap objects.
    norm : SymLogNorm
        SymLogNorm object to be used with the new colormap.
    total_points : int, optional
        Total number of points in the new colormap. The default is 2000.

    Returns
    -------
    interp_cmap : ListedColormap
        Interpolated colormap to be used with the SymLogNorm object.
    """
    # Interpolate for the first half and second half of the colormap
    try:
        colors = o_cmap.colors
    except AttributeError:
        temp_colors = np.linspace(0, 1, total_points)
        colors = o_cmap(temp_colors)
    len_cmap = len(colors)
    half_len = len_cmap // 2 + 1

    loc_zero = norm(0)

    n_low = int(loc_zero * total_points)
    n_high = total_points - n_low

    h1_cmap = get_cmap_interp(colors[:half_len], num_points=n_low)
    h2_cmap = get_cmap_interp(colors[half_len:], num_points=n_high)

    # Combine the two halves
    interp_cmap = h1_cmap + h2_cmap

    # Create a ListedColormap object from the interpolated colormap
    interp_cmap = ListedColormap(interp_cmap)

    return interp_cmap


def get_log_cmap(o_cmap, norm, total_points=2000):
    """
    Warps the upper half of a symmetric colormap for use with LogNorm.
    """
    # Sample the upper half of the original colormap
    try:
        base_vals = np.linspace(0.5, 1.0, total_points)
        colors = o_cmap(base_vals)
    except AttributeError:
        raise ValueError("Expected a Colormap object")

    # Warp the color positions logarithmically
    log_vals = np.logspace(np.log10(norm.vmin), np.log10(norm.vmax), total_points)
    norm_vals = norm(log_vals)  # These are values between 0 and 1
    warped_colors = o_cmap(0.5 + norm_vals * 0.5)  # Scale to [0.5, 1.0] range

    interp_colors = get_cmap_interp(warped_colors, num_points=total_points)
    return ListedColormap(interp_colors)


def get_linear_cmap(o_cmap, norm: Normalize, total_points=2000):
    """
    Create a linear colormap with more detail near zero, using a Normalize object.
    
    Parameters
    ----------
    o_cmap : Colormap
        Original colormap (e.g., from plt.cm).
    norm : Normalize
        Normalize instance with vmin/vmax defining the range.
    total_points : int
        Number of output color points.

    Returns
    -------
    ListedColormap
        The final interpolated colormap with enhanced resolution near zero.
    """
    loc_0 = norm(0.0)

    if loc_0 <= 0:
        # All data is positive: use top half of colormap
        sample_vals = np.linspace(0.5, 1.0, total_points)
        colors = o_cmap(sample_vals)
    elif loc_0 >= 1:
        # All data is negative: use bottom half of colormap
        sample_vals = np.linspace(0.0, 0.5, total_points)
        colors = o_cmap(sample_vals)
    else:
        # Mixed-sign data: split around 0, ensuring that 0 maps to cmap(0.5)
        n_low = int(loc_0 * total_points)
        n_high = total_points - n_low

        # Ensure zero at center
        vals_low = np.linspace(norm.vmin, 0, n_low, endpoint=False)
        vals_high = np.linspace(0, norm.vmax, n_high)
        sample_vals = np.concatenate([vals_low, vals_high])
        normed_vals = norm(sample_vals)

        # Shift normed_vals so that norm(0) maps to 0.5
        shift = 0.5 - norm(0.0)
        normed_vals = np.clip(normed_vals + shift, 0.0, 1.0)

        colors = o_cmap(normed_vals)

    # Ensure shape is (N, 3)
    colors = np.asarray(colors)[:, :3]

    interp_colors = get_cmap_interp(colors, num_points=total_points)
    return ListedColormap(interp_colors)
