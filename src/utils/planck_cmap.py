from pathlib import Path

from matplotlib.colors import ListedColormap
import numpy as np


colombi1_cmap = ListedColormap(np.loadtxt(Path(__file__).parent / "planck_colormap.txt")/255.)
colombi1_cmap.set_bad("gray") # color of missing pixels
colombi1_cmap.set_under("white") # color of background, necessary if you want to use
