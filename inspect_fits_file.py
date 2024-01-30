import healpy as hp
from astropy.io import fits
import matplotlib.pyplot as plt
from pathlib import Path
# from planck_instrument import map_files


def print_out_header(fits_fn):
    # Open the FITS file
    with fits.open(fits_fn) as hdul:
        # Loop over all HDUs in the FITS file
        for i, hdu in enumerate(hdul):
            print(f"Header for HDU {i}:")
            for card in hdu.header.cards:
                print(f"{card.keyword}: {card.value}")
            print("\n" + "-"*50 + "\n")


def get_num_fields(fits_fn):
    # Open the FITS file
    n_fields = {}
    with fits.open(fits_fn) as hdul:
        # Loop over all HDUs in the FITS file
        for i, hdu in enumerate(hdul):
            if i == 0:
                continue
            # print(f"Header for HDU {i}:")
            for card in hdu.header.cards:
                if card.keyword == "TFIELDS":
                    n_fields[i] = card.value
                # print(f"{card.keyword}: {card.value}")
            # print("\n" + "-"*50 + "\n")
    return n_fields


def get_fits_information(fits_fn):
    n_fields_per_hdu = get_num_fields(fits_fn)
    watch_keys = {}
    types_str_base = "TTYPE"
    units_str_base = "TUNIT"
    maps_info = {}
    for hdu_n in n_fields_per_hdu:
        indices = list(range(1, n_fields_per_hdu[hdu_n] + 1))
        field_types_keys = [f"{types_str_base}{i}" for i in indices]
        field_units_keys = [f"{units_str_base}{i}" for i in indices]
        watch_keys[hdu_n] = {"types": field_types_keys, "units": field_units_keys}

        maps_info[hdu_n] = {}
    with fits.open(fits_fn) as hdul:
        # header = hdul[1].header
        # hdr_dict = dict(header)

        # Loop over all HDUs in the FITS file
        for hdu_n, hdu in enumerate(hdul):
            if hdu_n == 0:
                continue
            maps_info[hdu_n] = dict(hdu.header)
            maps_info[hdu_n]["FIELDS"] = {}
            for field_n in range(1, n_fields_per_hdu[hdu_n] + 1):
                field_info = {}
                # Construct the keys for type and unit
                ttype_key = f'TTYPE{field_n}'
                tunit_key = f'TUNIT{field_n}'

                # Retrieve type and unit for the current field, if they exist
                field_info['type'] = maps_info[hdu_n].get(ttype_key, None)
                field_info['unit'] = maps_info[hdu_n].get(tunit_key, None)

                # Add the field info to the fields_info dictionary
                maps_info[hdu_n]["FIELDS"][field_n] = field_info
    return maps_info


def pretty_print_dict(this_dict):
    from pprint import pprint
    pprint(this_dict) 


def show_all_maps(fits_fn):
    n_fields_per_hdu = get_num_fields(fits_fn)
    map_info = get_fits_information(fits_fn)
    for hdu_n, n_fields in n_fields_per_hdu.items():
        # nested = map_info[hdu_n]["ORDERING"]  # nested is incorrect for the noise maps (maybe not others?)
        for field in range(n_fields):
            print(hdu_n, field)
            this_map = hp.read_map(fits_fn, hdu=hdu_n, field=field)
            hp.mollview(this_map)
            plt.show()


def show_one_map(fits_fn, hdu_n, field_n):
    this_map = hp.read_map(fits_fn, hdu=hdu_n, field=field_n)
    hp.mollview(this_map)
    plt.show()
  

def main():
    # noise_map_fn = "fidu_noise/ffp10_noise_030_full_map_mc_00000.fits"
    # petroff_used_fn = "ref_maps/HFI_SkyMap_100_2048_R3.01_full.fits"
    maybe_ok_fn = "planck_assets/LFI_SkyMap_070-BPassCorrected_1024_R3.00_full.fits"
    # maybe_ok_fn = "planck_assets/LFI_SkyMap_070-BPassCorrected_1024_R3.00_full.fits"
    # maybe_ok_fn = "planck_assets/HFI_SkyMap_100_2048_R3.01_full.fits"
    # maybe_ok_fn = "planck_assets/HFI_SkyMap_100-field-IQU_2048_R3.00_full.fits"

    if Path(maybe_ok_fn).exists():
        print("Exists!")
    else:
        print("No exists!")

    # for fn in map_files.values():
    #     print_out_header(fn)

    # show_one_map(petroff_used_fn, 1, 6)
    # print_out_header(noise_map_fn)
    # print_out_header(petroff_used_fn)
    # print(get_num_fields(noise_map_fn))
    # show_all_maps(noise_map_fn)

    # pretty_print_dict(get_fits_information(petroff_used_fn))
    pretty_print_dict(get_fits_information(maybe_ok_fn))

    show_all_maps(maybe_ok_fn)


if __name__ == "__main__":
    main()
