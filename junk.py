from utils.planck_cmap import colombi1_cmap
from utils.fits_inspection import get_num_fields

import healpy as hp
import matplotlib.pyplot as plt


def viz_map(path) -> None:
        fields_strs = "TQU"
        n_fields = get_num_fields(path)[1]  # Assume HDUL 1, which is valid because all maps are produced by us.
        for field_idx in range(n_fields):
            curr_field = fields_strs[field_idx]
            this_map = hp.read_map(path, field=field_idx)
            hp.mollview(this_map, cmap=colombi1_cmap)
            
            
        plt.show()
        plt.close()


pre_refactor_fid = '/Users/ammaraljerwi/research/jim_cmb_code/cmb_datasets/TQU_128/Raw/Dummy0/sim0000/cmb_map_fid.fits'
viz_map(pre_refactor_fid)

post_refactor_fid = '/Users/ammaraljerwi/research/jim_cmb_code/cmb_datasets/TQU_128_2_2/Simulations/Train/sim0000/cmb_map_fid.fits'
viz_map(post_refactor_fid)
