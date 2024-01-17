from time import time
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import healpy as hp
import pysm3
import pysm3.units as u
from cmb_component import write_cls
from planck_instrument import PlanckInstrument, PlanckDetector
from planck_cmap import colombi1_cmap
from urllib.error import URLError


# Dev path  1 goal: Show simplest map
# Dev path  2 goal: Show skymap with simple features
# Dev path  3 goal: Show skymaps with recommended features:
# https://galsci.github.io/blog/2022/common-fiducial-sky/
# https://galsci.github.io/blog/2022/common-fiducial-extragalactic-cmb/
# Dev path  4 goal: Show skymaps with simplest possible noise implementation
# Dev path  5 goal: Skymaps with simple noise and simple beam convolution
# Dev path  6 goal: Skymaps of CMB preset strings
#                   Also, break out TQU maps instead of commenting out lines 
# Dev path  7 goal: Skymaps of CMB with seeding
# Dev path  8 goal: Better beam convolution
# Dev path  9 goal: Better instrumentation noise
# Dev path 10 goal: Good CMB, good beam convolution, good instrumentation noise, all foregrounds
#                   Baseline for map production

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

# https://galsci.github.io/blog/2022/common-fiducial-sky/
low_complexity_foregrounds = ["d9","s4","f1","a1","co1"]
medium_complexity_foregrounds = ["d10","s5","f1","a1","co3"]
high_complexity_foregrounds = ["d12","s7","f1","a2","co3"]
# Extragalactic Foregrounds: https://galsci.github.io/blog/2022/common-fiducial-extragalactic-cmb/
# extragalactic_foregrounds = ["cib1", "tsz1", "ksz1"]
# extragalactic_foregrounds = []
extragalactic_foregrounds = ["cib1", "tsz1", "ksz1", "rg1"]

low_complexity_with_extragalactic_foregrounds = [
    *low_complexity_foregrounds,
    *extragalactic_foregrounds
]

def simulate_sky(output_dir="out", show_renders=True, save_renders=False):
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(exist_ok=True, parents=True)
    nside = 512
    ellmax = 8150

    planck = PlanckInstrument(nside)
    rng = np.random.default_rng(seed=8675309)

    complexities = dict(
        # low=low_complexity_foregrounds,
        lowplus=low_complexity_with_extragalactic_foregrounds,  # only for downloading data
        medium=medium_complexity_foregrounds,
        high=high_complexity_foregrounds
    )

    for complexity_lbl, complexity_fgs in complexities.items():
        cmb_ps_filename = write_cls(ellmax=ellmax)
        cmb = pysm3.CMBLensed(nside=nside, 
                              cmb_spectra=cmb_ps_filename, 
                              cmb_seed=0, 
                              apply_delens=False)

        # For use with limited download everything setup
        # TODO: Implement this better (hydra?)
        sky = pysm3.Sky(nside=nside, 
                        preset_strings=complexity_fgs,
                        component_objects=[cmb])
        # sky = pysm3.Sky(nside=nside, 
        #                 preset_strings=[
        #                     *complexity_fgs,
        #                     *extragalactic_foregrounds], 
        #                 component_objects=[cmb])

        field_strs = ["T", "Q", "U"]
        # field_strs = ["T"]

        # Usual Planck ranges for maps
        val_ranges = [dict(min=-300, max=300),
                      dict(min=-2.5, max=2.5),
                      dict(min=-2.5, max=2.5),]

        # Extended range for QU; one discussion spoke of doubleing variance, not effective
        # val_ranges = [dict(min=-300, max=300),
        #               dict(min=-5, max=5),
        #               dict(min=-5, max=5),]

        # No range for QU maps; we have values in the 100's
        # val_ranges = [dict(min=-300, max=300),
        #               dict(),
        #               dict(),]

        # planck_freqs = [30]
        planck_freqs = [30, 44, 70, 100, 143, 217, 353, 545, 857]
        # planck_freqs = [545, 857]
        for nominal_freq in planck_freqs:
            print(nominal_freq)
            detector = planck.detectors[nominal_freq]
            frequency = detector.center_frequency
            try:
                skymaps = sky.get_emission(frequency)
            except URLError as e:
                print(f"While working on {complexity_lbl}, error in detector {nominal_freq}.")
                print(e)
                continue
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
                                title=f"No  Noise; Complexity: {complexity_lbl}, Field: {field_str}, {nominal_freq} GHz", 
                                unit=final_map.unit,
                                cmap=colombi1_cmap)
                    hp.mollview(final_map, **val_range, 
                                title=f"Yes Noise; Complexity: {complexity_lbl}, Field: {field_str}, {nominal_freq} GHz", 
                                unit=final_map.unit,
                                cmap=colombi1_cmap)
                    plt.show()
                    plt.close()
                if save_renders:
                    hp.mollview(map_smoothed, **val_range, 
                                title=f"No  Noise; Complexity: {complexity_lbl}, Field: {field_str}, {nominal_freq} GHz", 
                                unit=final_map.unit,
                                cmap=colombi1_cmap)
                    plt.savefig(f"{output_dir}/cmb_map_{complexity_lbl}_{i}_{nominal_freq}_{field_str}_no_noise_cf.png")
                    plt.clf()
                    hp.mollview(final_map, **val_range, 
                                title=f"Yes Noise; Complexity: {complexity_lbl}, Field: {field_str}, {nominal_freq} GHz", 
                                unit=final_map.unit,
                                cmap=colombi1_cmap)
                    plt.savefig(f"{output_dir}/cmb_map_{complexity_lbl}_{i}_{nominal_freq}_{field_str}_yes_noise_cf.png")
                    plt.clf()
                    plt.close()
                i+=1


if __name__ == "__main__":
    simulate_sky(show_renders=False, save_renders=False)
    # simulate_sky(show_renders=True, save_renders=False)
    # simulate_sky()
