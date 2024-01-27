import numpy as np
import pysm3
import pysm3.units as u
import matplotlib.pyplot as plt
import healpy as hp
from time import time
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from planck_cmap import colombi1_cmap
from cmb_component_old import write_cls
from pathlib import Path


# Dev path 1 goal: Show simplest map
# Dev path 2 goal: Show skymap with simple features
# Dev path 3 goal: Show skymaps with recommended features:
# https://galsci.github.io/blog/2022/common-fiducial-sky/
# https://galsci.github.io/blog/2022/common-fiducial-extragalactic-cmb/
# Dev path 4 goal: Show skymaps with simplest possible noise implementation
# Dev path 5 goal: Skymaps with simple noise and simple beam convolution
# Dev path 6 goal: Skymaps of CMB preset strings
#                  Also, break out TQU maps instead of commenting out lines 
# Dev path 7 goal: Skymaps of CMB with seeding

"""
c#:    [1- 4]  cmb
d#:    [1-12]  dust
s#:    [1- 7]  synchrotron
a#:    [1- 2]  ame
f#:    [1]     free-free
co#:   [1- 3]  CO line Emission
cib#:  [1]     Cosmic Infrared Background
tsz#:  [1]     Thermal Sunyaev-Zeldovich
ksz#:  [1]     Kinetic Sunyaev-Zeldovich
rg#:   [1]     Radio Galaxies
"""

def simulate_sky(output_dir="out"):
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(exist_ok=True, parents=True)
    colombi1_cmap = ListedColormap(np.loadtxt("planck_colormap.txt")/255.)
    colombi1_cmap.set_bad("gray") # color of missing pixels
    colombi1_cmap.set_under("white") # color of background, necessary if you want to use
    
    nside = 512
    ellmax = 8150

    # options = ["Syn"]
    # options = ["c1", "c2", "c3", "c4"]
    options = ["c1", "c2", "c3", "c4", "Syn"]

    for option in tqdm(options):
        if option == "Syn":
            filename = write_cls(ellmax=ellmax)
            cmb = pysm3.CMBLensed(nside=nside, 
                                  cmb_spectra=filename, 
                                  cmb_seed=0, 
                                  apply_delens=False)
            sky = pysm3.Sky(nside=nside, component_objects=[cmb])

        else:
            sky = pysm3.Sky(nside=nside, preset_strings=[option])

        field_strs = ["T", "Q", "U"]
        val_ranges = [dict(min=-300, max=300),
                    dict(min=-2.5, max=2.5),
                    dict(min=-2.5, max=2.5),]

        # planck_freqs = [30, 44, 70, 100, 143, 217, 353, 545, 857]
        planck_freqs = [100]
        for freq in planck_freqs:
            frequency = freq * u.GHz
            skymaps = sky.get_emission(frequency)
            fwhm_degrees = 0.16*u.deg  # Full width at half maximum in degrees

            i = 1
            for skymap, field_str, val_range in zip(skymaps, field_strs, val_ranges):
                map_smoothed = pysm3.apply_smoothing_and_coord_transform(
                    skymap, fwhm=fwhm_degrees)
                # map_smoothed = hp.smoothing(skymap, fwhm=np.radians(fwhm_degrees))
                hp.mollview(map_smoothed, **val_range, 
                            title=f"Model: {option}, Field: {field_str}, {freq} GHz", 
                            unit=skymap.unit,
                            cmap=colombi1_cmap)
                # plt.show()
                plt.savefig(f"{output_dir}/cmb_map_{option}_{i}_{field_str}_{freq}.png")
                plt.clf()
                i+=1


if __name__ == "__main__":
    simulate_sky()
