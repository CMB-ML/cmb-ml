import numpy as np
import pysm3
import pysm3.units as u
import matplotlib.pyplot as plt
import healpy as hp
from time import time


# Dev path 1 goal: Show simplest map
# Dev path 2 goal: Show skymap with simple features
# Dev path 3 goal: Show skymaps with recommended features:
# https://galsci.github.io/blog/2022/common-fiducial-sky/
# https://galsci.github.io/blog/2022/common-fiducial-extragalactic-cmb/
# Dev path 4 goal: Show skymaps with simplest possible noise implementation


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

def simulate_sky():
    nside = 128
    sky = pysm3.Sky(nside=nside, preset_strings=["d9", "s4", "f1", "a1", "co1"])

    # planck_freqs = [30, 44, 70, 100, 143, 217, 353, 545, 857]
    planck_freqs = [100]

    for freq in planck_freqs:
        frequency = freq * u.GHz
        skymap = sky.get_emission(frequency)

        # Add Gaussian noise
        npix = hp.nside2npix(nside)
        noise_std = 1e-5  # Adjust this value for noise level
        noise = np.random.normal(0, noise_std, (3, npix)) * skymap.unit
        map_with_noise = skymap + noise

        hp.mollview(map_with_noise[0], title=f"Temperature Map, Frequency {freq} GHz", unit=map_with_noise.unit)
        plt.show()
        # hp.mollview(map_with_noise[1], title=f"Polarization Q Map, Frequency {freq} GHz", unit=map_with_noise.unit)
        # plt.show()
        # hp.mollview(map_with_noise[2], title=f"Polarization U Map, Frequency {freq} GHz", unit=map_with_noise.unit)
        # plt.show()

if __name__ == "__main__":
    simulate_sky()
