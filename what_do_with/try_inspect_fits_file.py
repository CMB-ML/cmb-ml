import healpy as hp
from astropy.io import fits
import matplotlib.pyplot as plt
from pathlib import Path
from utils.fits_inspection import (print_out_header,
                                   get_num_fields,
                                   get_fits_information,
                                   print_fits_information,
                                   show_all_maps,
                                   show_one_map)
  

def main():
    maybe_ok_fn = "planck_assets/LFI_SkyMap_070-BPassCorrected_1024_R3.00_full.fits"
    if Path(maybe_ok_fn).exists():
        print("Exists!")
    else:
        print("No exists!")

    print_out_header(maybe_ok_fn)
    print_fits_information(maybe_ok_fn)
    show_all_maps(maybe_ok_fn)


if __name__ == "__main__":
    main()
