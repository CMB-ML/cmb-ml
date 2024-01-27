from time import time
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import healpy as hp
import pysm3
import pysm3.units as u
from cmb_component_old import write_cls
from planck_instrument_old import PlanckInstrument
from planck_cmap import colombi1_cmap


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
# Dev path 9 goal: Better instrumentation noise

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

def simulate_sky(output_dir="out", show_renders=True, save_renders=False):
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(exist_ok=True, parents=True)
    nside = 512
    ellmax = 8150

    # options = ["Syn"]
    options = ["c1"]
    # options = ["c1", "c2", "c3", "c4"]
    # options = ["c1", "c2", "c3", "c4", "Syn"]

    # field_strs = ["T", "Q", "U"]
    field_strs = ["T"]
    val_ranges = [
                  dict(min=-300, max=300),
                  dict(min=-2.5, max=2.5),
                  dict(min=-2.5, max=2.5)
                  ]

    # planck_freqs = [30, 44, 70, 100, 143, 217, 353, 545, 857]
    planck_freqs = [100, 217, 545, 857]

    planck = PlanckInstrument(nside)
    rng = np.random.default_rng(seed=8675309)

    # for option in tqdm(options):
    for option in options:
        if option == "Syn":
            filename = write_cls(ellmax=ellmax)
            cmb = pysm3.CMBLensed(nside=nside, 
                                  cmb_spectra=filename, 
                                  cmb_seed=0, 
                                  apply_delens=False)
            sky = pysm3.Sky(nside=nside, component_objects=[cmb], output_unit="uK_RJ")
        else:
            sky = pysm3.Sky(nside=nside, preset_strings=[option], output_unit="uK_RJ")

        for nominal_freq in planck_freqs:
            print(nominal_freq)
            detector = planck.detectors[nominal_freq]
            frequency = detector.center_frequency
            skymaps = sky.get_emission(frequency)
            fwhm = detector.fwhm

            i = 1
            for skymap, field_str, val_range in zip(skymaps, field_strs, val_ranges):
                if nominal_freq in [545, 857] and field_str != "T":
                    continue  # 545 and 857 do not have polarization maps
                noise_map = detector.get_noise_map(field_str, rng, force_overwrite=True)
                map_smoothed = pysm3.apply_smoothing_and_coord_transform(skymap, fwhm=fwhm)
                final_map = map_smoothed + noise_map
                if show_renders:
                    hp.mollview(map_smoothed, **val_range, 
                                title=f"No  Noise Model: {option}, Field: {field_str}, {nominal_freq} GHz", 
                                unit=final_map.unit,
                                cmap=colombi1_cmap)
                    hp.mollview(final_map, **val_range, 
                                title=f"Yes Noise Model: {option}, Field: {field_str}, {nominal_freq} GHz", 
                                unit=final_map.unit,
                                cmap=colombi1_cmap)
                    plt.show()
                    plt.close()
                if save_renders:
                    hp.mollview(map_smoothed, **val_range, 
                                title=f"No  Noise Model: {option}, Field: {field_str}, {nominal_freq} GHz", 
                                unit=final_map.unit,
                                cmap=colombi1_cmap)
                    plt.savefig(f"{output_dir}/cmb_map_{option}_{i}_{field_str}_{nominal_freq}_no_noise_cf.png")
                    plt.clf()
                    hp.mollview(final_map, **val_range, 
                                title=f"Yes Noise Model: {option}, Field: {field_str}, {nominal_freq} GHz", 
                                unit=final_map.unit,
                                cmap=colombi1_cmap)
                    plt.savefig(f"{output_dir}/cmb_map_{option}_{i}_{field_str}_{nominal_freq}_yes_noise_cf.png")
                    plt.clf()
                    plt.close()
                i+=1


if __name__ == "__main__":
    # simulate_sky(show_renders=False, save_renders=True)
    simulate_sky(show_renders=True, save_renders=False, output_dir="out6")
    # simulate_sky()
