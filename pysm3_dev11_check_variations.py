import numpy as np
import pysm3
from planck_instrument import PlanckInstrument
from urllib.error import URLError
from itertools import product
from http.client import RemoteDisconnected
import json


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
# Dev path 11 goal: Check variations in all preset strings
#                   Result: no variations

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
extragalactic_foregrounds = ["cib1", "tsz1", "ksz1", "rg1"]

all_preset_strings = [
    *low_complexity_foregrounds,
    *medium_complexity_foregrounds,
    *high_complexity_foregrounds,
    *extragalactic_foregrounds
]


def get_full_class_name(obj):
    # From https://stackoverflow.com/a/58045927
    module = obj.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return obj.__class__.__name__
    return module + '.' + obj.__class__.__name__

def simulate_sky():
    nside = 512

    planck = PlanckInstrument(nside)
    planck_freqs = [30, 44, 70, 100, 143, 217, 353, 545, 857]
    field_strs = ["T", "Q", "U"]

    combinations = list(product(planck_freqs, all_preset_strings))

    results = {}

    for nominal_freq, preset_string in combinations:

        sky1 = pysm3.Sky(nside=nside, 
                        preset_strings=[preset_string],
                        output_unit="uK_CMB")
        sky2 = pysm3.Sky(nside=nside, 
                        preset_strings=[preset_string],
                        output_unit="uK_CMB")

        print(nominal_freq, preset_string)
        if nominal_freq not in results.keys():
            results[nominal_freq] = {}

        detector = planck.detectors[nominal_freq]
        frequency = detector.center_frequency
        try:
            skymaps1 = sky1.get_emission(frequency)
            skymaps2 = sky2.get_emission(frequency)
        except URLError as e:
            print(e)
            results[nominal_freq][preset_string] = "Fail to execute, URLError"
            continue
        except RemoteDisconnected as e:
            print(e)
            results[nominal_freq][preset_string] = "Fail to execute, RemoteDisconnected"
            continue
        except Exception as e:
            print(e)
            # TODO: Fix this kludge, test and figure out how to actually handle it.
            try:
                results[nominal_freq][preset_string] = f"Fail to execute, other error: {get_full_class_name(e)}"
            except Exception as e2:
                print(e2)
                results[nominal_freq][preset_string] = f"Fail to execute, other error: {e}, also caused {e2}"
            continue

        results[nominal_freq][preset_string] = {}
        fwhm = detector.fwhm

        for skymap1, skymap2, field_str in zip(skymaps1, skymaps2, field_strs):
            map_smoothed1 = pysm3.apply_smoothing_and_coord_transform(skymap1, fwhm=fwhm)
            map_smoothed2 = pysm3.apply_smoothing_and_coord_transform(skymap2, fwhm=fwhm)

            diff = map_smoothed1 - map_smoothed2

            max_diff = np.max(diff)

            results[nominal_freq][preset_string][field_str] = str(max_diff)

            if max_diff > 0:
                print(f"Non-zero difference found for {field_str}, {max_diff}")
    result_str = json.dumps(results, indent=2, sort_keys=True)
    print(result_str)
    with open('check_variation_result.txt', 'w', encoding='utf8') as f:
        json.dump(results, f, sort_keys=True, indent=2)


if __name__ == "__main__":
    simulate_sky()
