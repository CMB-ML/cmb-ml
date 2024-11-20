import numpy as np
from scipy.interpolate import interp1d

from matplotlib.colors import SymLogNorm, ListedColormap


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
