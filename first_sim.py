import numpy as np
import pysm3
import pysm3.units as u
import matplotlib.pyplot as plt
import healpy as hp
from time import time


# Map synthesis 0 (basic make map using PySM)

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
    start_time = time()
    sky = pysm3.Sky(nside=nside, preset_strings=["d9", "s4", "f1", "a1", "co1"])
    map = sky.get_emission(100 * u.GHz)
    print(f"Time for simple sky: {time() - start_time:.1f}")
    # start_time = time()
    # sky = pysm3.Sky(nside=nside, preset_strings=["d10", "s5", "f1", "a1", "co3"])
    # map = sky.get_emission(100 * u.GHz)
    # print(f"{time() - start_time:.1f}")
    # start_time = time()
    # sky = pysm3.Sky(nside=nside, preset_strings=["d12", "s7", "f1", "a2", "co3"])
    # map = sky.get_emission(100 * u.GHz)
    # print(f"{time() - start_time:.1f}")
    # start_time = time()
    # sky = pysm3.Sky(nside=nside, preset_strings=["cib1", "ksz1", "tsz1", "rg1"])
    # map = sky.get_emission(100 * u.GHz)
    # print(f"{time() - start_time:.1f}")
    # start_time = time()

    # planck_freqs = [30, 44, 70, 100, 143, 217, 353, 545, 857]
    planck_freqs = [100]

    for freq in planck_freqs:
        frequency = freq * u.GHz
        map = sky.get_emission(frequency)

        # Add Gaussian noise
        npix = hp.nside2npix(nside)
        noise_std = 1e-5  # Adjust this value for noise level
        noise = np.random.normal(0, noise_std, (3, npix)) * map.unit
        map_with_noise = map + noise

        hp.mollview(map_with_noise[0], title=f"Frequency {freq} GHz", unit=map_with_noise.unit)
        plt.show()
        # hp.mollview(map_with_noise[1], title=f"Frequency {freq} GHz - Free-Free", unit=map_with_noise.unit)
        # plt.show()
        # hp.mollview(map_with_noise[2], title=f"Frequency {freq} GHz - CMB", unit=map_with_noise.unit)
        # plt.show()

if __name__ == "__main__":
    simulate_sky()
