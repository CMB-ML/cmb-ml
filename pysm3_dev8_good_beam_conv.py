import numpy as np
import pysm3
import pysm3.units as u
import matplotlib.pyplot as plt
import healpy as hp
from time import time
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from planck_cmap import colombi1_cmap
from cmb_component import write_cls
from planck_instrument import PlanckInstrument


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
# Dev path 8 goal: Better beam convolution

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
    nside = 512

    options = ["c1"]

    planck = PlanckInstrument(nside)

    for option in tqdm(options):
        if option == "Syn":
            pass
        else:
            sky = pysm3.Sky(nside=nside, preset_strings=[option])

        field_strs = ["T", "Q", "U"]
        val_ranges = [dict(min=-300, max=300),
                      dict(min=-2.5, max=2.5),
                      dict(min=-2.5, max=2.5),]

        # planck_freqs = [30, 44, 70, 100, 143, 217, 353, 545, 857]
        planck_freqs = [30]
        for freq in planck_freqs:
            frequency = freq * u.GHz
            skymaps = sky.get_emission(frequency)
            fwhm = planck.detectors[freq].fwhm

            i = 1
            for skymap, field_str, val_range in zip(skymaps, field_strs, val_ranges):
                map_smoothed = pysm3.apply_smoothing_and_coord_transform(
                    skymap, fwhm=fwhm)
                hp.mollview(map_smoothed, **val_range, 
                            title=f"Model: {option}, Field: {field_str}, {freq} GHz", 
                            unit=skymap.unit,
                            cmap=colombi1_cmap)
                plt.show()
                plt.savefig(f"{output_dir}/cmb_map_{option}_{i}_{field_str}_{freq}.png")
                plt.clf()
                i+=1


if __name__ == "__main__":
    simulate_sky()
