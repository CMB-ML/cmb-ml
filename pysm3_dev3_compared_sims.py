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
    print("Starting Low Complexity. Takes 86s on Lab Workstation.")
    start_time = time()
    # Low complexity contaminants + extragalactic
    sky = pysm3.Sky(nside=nside, preset_strings=["c1", "d9", "s4", "f1", "a1", "co1", "cib1", "ksz1", "tsz1", "rg1"])
    skymap = sky.get_emission(100 * u.GHz)
    print(f"Time for low complexity sky: {time() - start_time:.1f}")
    print("Starting Medium Complexity. Takes 86s on Lab Workstation.")  # No typo, same as Low Complexity
    start_time = time()
    # Medium complexity contaminants + extragalactic
    sky = pysm3.Sky(nside=nside, preset_strings=["c1", "d10", "s5", "f1", "a1", "co3", "cib1", "ksz1", "tsz1", "rg1"])
    skymap = sky.get_emission(100 * u.GHz)
    print(f"Time for medium complexity sky: {time() - start_time:.1f}")
    print("Starting High Complexity. Takes 152s on Lab Workstation.")
    start_time = time()
    # High complexity contaminants + extragalactic
    sky = pysm3.Sky(nside=nside, preset_strings=["c1", "d12", "s7", "f1", "a2", "co3", "cib1", "ksz1", "tsz1", "rg1"])
    skymap = sky.get_emission(100 * u.GHz)
    print(f"Time for high complexity sky: {time() - start_time:.1f}")


if __name__ == "__main__":
    simulate_sky()
