import numpy as np
import healpy as hp


def get_power(mapp1,mapp2=None,mask=None,lmax=None, use_pixel_weights=False):
    if mask is not None:
        # if mapp2 is None:
        #     mapp2 = mapp1
        mean1 = np.sum(mapp1*mask)/np.sum(mask)
        mean2 = np.sum(mapp2*mask)/np.sum(mask)
        fsky = np.sum(mask)/mask.shape[0]
        ps = hp.anafast(mask*(mapp1-mean1),mask*(mapp2-mean2),lmax=lmax, use_pixel_weights=use_pixel_weights)/fsky
    else:
        ps = hp.anafast(mapp1, mapp2, lmax=lmax, use_pixel_weights=use_pixel_weights)
    return ps
