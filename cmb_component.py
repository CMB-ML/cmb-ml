import os
import numpy as np
import pysm3
import healpy as hp
import camb
import matplotlib.pyplot as plt
import pysm3.units as u


# Based on https://camb.readthedocs.io/en/latest/CAMBdemo.html


def get_cls(ellmax):
    #Set up a new set of parameters for CAMB
    pars = camb.CAMBparams()
    #This function sets up with one massive neutrino and helium set using BBN consistency
    pars.set_cosmology(H0=67.4, ombh2=0.0224, omch2=0.120, mnu=0.06, omk=0.001, tau=0.054)
    pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
    pars.set_for_lmax(ellmax, lens_potential_accuracy=0)

    results = camb.get_results(pars)

    # TODO: Ensure correct CMB_unit
    # We want raw_cl: this is unnormalized C_ell; the alternative is normalized D_ell
    # TODO: USing 'total' spectra
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', spectra=('total',), raw_cl=True)

    totCL=powers['total']

    return totCL


def write_cls(ellmax):
    filename = "cmb_spectrum.txt"
    #Set up a new set of parameters for CAMB
    pars = camb.CAMBparams()
    #This function sets up with one massive neutrino and helium set using BBN consistency
    pars.set_cosmology(H0=67.4, ombh2=0.0224, omch2=0.120, mnu=0.06, omk=0.001, tau=0.054)
    pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
    pars.set_for_lmax(ellmax, lens_potential_accuracy=0)

    results = camb.get_results(pars)
    results.save_cmb_power_spectra(filename=filename)

    return filename


# class CMBMapFromCl(pysm3.Model):
#     """Synthesizes CMB maps from input Cls"""

#     def __init__(
#         self, 
#         nside, 
#         cl: np.ndarray,
#         seed: int,
#         max_nside=None,
#         map_dist=None
#     ):
#         """
#         Class template source is PySM3 CMBMap
#         The input is assumed to be in `uK_CMB`

#         Parameters
#         ----------
#         nside: int
#             HEALPix N_side parameter of the maps to create
#         cls: `np.ndarray` object
#             The cls to use for creating a CMB realization.
#             Expects an array shape (maxell, 4), in the order TT, 
#         seed: int
#             healpy changes numpy's random seed in order to control realizations
#         max_nside: int
#             Keeps track of the the maximum Nside this model is available at
#             by default 512 like PySM 2 models. See 
#             https://pysm3.readthedocs.io/en/latest/index.html#best-practices-for-model-execution
#             Unsure how to set this; leaving as-is for now.
#         map_IQU: `pathlib.Path` object
#             Path to a single IQU map
#         """
#         # TODO: Set max_nside to reasonable value?
#         # TODO: Figure out how to do MPI stuff with hp synalm and alm2map (?)
#         if max_nside is None:
#             max_nside = nside
#         super().__init__(nside=nside, max_nside=max_nside, map_dist=map_dist)
#         if not isinstance(cl, np.ndarray):
#             raise TypeError("cl must be a numpy array, shape (maxell, 4)")
#         if cl.shape[1] != 4:
#             raise ValueError("cl must be a numpy array shape (maxell, 4)")
#         self.cl = cl
#         self.seed = seed
#         self.map = CMBMapFromCl.make_map(cl, seed, nside)

#     @staticmethod
#     def make_map(cl, seed, nside):
#         # TODO: Fix healpy seed method (somehow?) (update healpy's function?)
#         np.random.seed(seed)
        
#         # TODO: Make synalm flexible, less python is better (update healpy's function?)
#         # Convert to a list of four numpy arrays
#         cl = list(cl.T)

#         alm = hp.synalm(cl, new=False)
#         cmb_map = hp.alm2map(alm, nside=nside)
#         # TODO: Is unit=u.uK_CMB correct? It's a guess based on CMBMap
#         return u.Quantity(cmb_map, unit=u.uK_CMB)

#     @u.quantity_input
#     def get_emission(self, freqs: u.GHz, weights=None) -> u.uK_RJ:
#         freqs = pysm3.utils.check_freq_input(freqs)
#         # do not use normalize weights because that includes a transformation
#         # to spectral radiance and then back to RJ
#         if weights is None:
#             weights = np.ones(len(freqs), dtype=np.float32)

#         scaling_factor = pysm3.utils.bandpass_unit_conversion(
#             freqs * u.GHz, weights, output_unit=u.uK_RJ, input_unit=u.uK_CMB
#         )

#         # a = self.map * scaling_factor
#         # b = u.Quantity(a, unit=u.uK_RJ, copy=False)

#         return u.Quantity(self.map * scaling_factor, unit=u.uK_RJ, copy=False)


# def main():
#     cl = get_cls()
#     cmb_map = make_map(cl, 512)
#     print(cmb_map.shape)
#     hp.mollview(cmb_map[0])
#     plt.show()


# if __name__ == "__main__":
#     main()
