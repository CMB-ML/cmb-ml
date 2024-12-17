import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ORIGINAL USED IN B_DISPLAY_TEST_PATCH_WHATEVER
# def show_patch(cmb_map, obs_maps, patch_pixels, title):
#     fig = plt.figure(figsize=(12, 9))  # Adjust the figure size as needed
#     gs = gridspec.GridSpec(3, 4, figure=fig, wspace=0.4, hspace=0.4)  # Adjust spacing
    
#     # Plot the first map
#     ax0 = fig.add_subplot(gs[0, 0])
#     im = ax0.imshow(cmb_map[patch_pixels].value, origin='lower', cmap='viridis')
#     ax0.set_title("CMB Map")
#     ax0.set_xticks([])
#     ax0.set_yticks([])
#     plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    
#     # Plot the remaining 9 maps
#     for freq_id, obs_map in enumerate(obs_maps):
#         row = (freq_id + 1) // 4  # Row index (skip the first slot)
#         col = (freq_id + 1) % 4  # Column index
#         ax = fig.add_subplot(gs[row, col])
#         im = ax.imshow(obs_map[patch_pixels].value, origin='lower', cmap='viridis')
#         ax.set_title(f"Obs Map {freq_id + 1}")
#         ax.set_xticks([])
#         ax.set_yticks([])
#         plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
#     plt.suptitle(title)
#     plt.show()



def show_patch(cmb_map, obs_maps, title) -> None:
    """
    Draws a 3x4 grid of maps; the first is the CMB, the rest are the observed maps.

    Args:
        cmb_map: The CMB map
        obs_maps: A list of observed maps (expects a list of 2D numpy arrays)
        title: The title of the plot

    Returns:
        None
    """
    fig = plt.figure(figsize=(12, 9))  # Adjust the figure size as needed
    gs = gridspec.GridSpec(3, 4, figure=fig, wspace=0.4, hspace=0.4)  # Adjust spacing
    
    # Plot the first map
    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(cmb_map, origin='lower', cmap='viridis')
    ax0.set_title("CMB Map")
    ax0.set_xticks([])
    ax0.set_yticks([])
    plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    
    # Plot the remaining 9 maps
    for freq_id, obs_map in enumerate(obs_maps):
        row = (freq_id + 1) // 4  # Row index (skip the first slot)
        col = (freq_id + 1) % 4  # Column index
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(obs_map, origin='lower', cmap='viridis')
        ax.set_title(f"Obs Map {freq_id + 1}")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle(title)
    plt.show()
