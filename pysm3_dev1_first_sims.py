import numpy as np
import pysm3
import pysm3.units as u
import matplotlib.pyplot as plt
import healpy as hp
from time import time


# Dev path 1 goal: Show simplest map

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
    sky = pysm3.Sky(nside=nside, preset_strings=["d9"])
    skymap = sky.get_emission(100 * u.GHz)
    hp.mollview(skymap[0])
    plt.show()


if __name__ == "__main__":
    simulate_sky()
