import logging

import healpy as hp
from pysm3 import CMBLensed, CMBMap
import pysm3.units as u
from pysm3.models.cmb import simulate_tebp_correlated
from omegaconf.errors import ConfigAttributeError
from cmbml.core import Asset

import numpy as np


logger = logging.getLogger(__name__)


class CMBFactory:
    """
    Factory for returning objects with the CMB component.
    Returns a basic CMB map or a lensed CMB map.
    Lensed is default by virtue of 'make_cmb()'.

    Attributes:
        nside (int): The nside for the sky.
        max_nside_pysm_component (int): The maximum nside for the PySM component.
        apply_delens (bool): Whether to apply delensing.
        delensing_ells (str): The delensing ells.
        map_dist (str): The map distribution for the CMB.
    
    Methods:
        make_basic_cmb(seed: int, cmb_ps_fid_path: str) -> CMBMap:
            Returns a basic CMB map.
        make_cmb(seed: int, cmb_ps_fid_path: str) -> CMBLensed:
            Returns a lensed CMB
    """
    def __init__(self, nside_sky):
        self.nside = nside_sky
        self.max_nside_pysm_component = None
        self.apply_delens = False
        self.delensing_ells = None
        self.map_dist = None

    def make_basic_cmb(self, seed, cmb_ps_fid_path) -> CMBMap:
        return BasicCMB(nside=self.nside,
                        cmb_spectra=cmb_ps_fid_path,
                        cmb_seed=seed,
                        max_nside=self.max_nside_pysm_component,
                        map_dist=self.map_dist)

    def make_cmb(self, seed, cmb_ps_fid_path) -> CMBLensed:
        return CMBLensed(nside=self.nside,
                         cmb_spectra=cmb_ps_fid_path,
                         cmb_seed=seed,
                         max_nside=self.max_nside_pysm_component,
                         apply_delens=self.apply_delens,
                         delensing_ells=self.delensing_ells,
                         map_dist=self.map_dist)


class BasicCMB(CMBMap):
    """
    Basic CMB
    Pulled from PySM3's CMBLensed class, with lensing removed.
    Correctness not guaranteed.
    
    Attributes:
        nside (int): The nside for the sky.
        cmb_spectra (str): The input text file from CAMB, spectra unlensed.
        cmb_seed (int): The numpy random seed for synfast, set to None for a random seed.
        map_dist (str): The map distribution.

    Methods:
        make_cmb() -> np.ndarray:
            Returns correlated CMB (T, Q, U) maps.
    """
    def __init__(
        self,
        nside,
        cmb_spectra,
        max_nside=None,
        cmb_seed=None,
        map_dist=None
        ):
        try:
            super().__init__(nside=nside, max_nside=max_nside, map_dist=map_dist)
        except ValueError:
            pass  # suppress exception about not providing any input map
        self.cmb_spectra = self.read_txt(cmb_spectra, unpack=True)
        self.cmb_seed = cmb_seed
        self.map = u.Quantity(self.make_cmb(), unit=u.uK_CMB, copy=False)

    def make_cmb(self):
        """Returns correlated CMB (T, Q, U) maps.

        :return: function -- CMB maps.
        """
        synlmax = 8 * self.nside  # this used to be user-defined.
        data = self.cmb_spectra
        lmax_cl = len(data[0]) + 1
        ell = np.arange(int(lmax_cl + 1))
        synlmax = min(synlmax, ell[-1])

        # Reading input spectra in CAMB format. CAMB outputs l(l+1)/2pi hence the corrections.
        cl_tt = np.zeros(lmax_cl + 1)
        cl_tt[2:] = 2 * np.pi * data[1] / (ell[2:] * (ell[2:] + 1))
        cl_ee = np.zeros(lmax_cl + 1)
        cl_ee[2:] = 2 * np.pi * data[2] / (ell[2:] * (ell[2:] + 1))
        cl_bb = np.zeros(lmax_cl + 1)
        cl_bb[2:] = 2 * np.pi * data[3] / (ell[2:] * (ell[2:] + 1))
        cl_te = np.zeros(lmax_cl + 1)
        cl_te[2:] = 2 * np.pi * data[4] / (ell[2:] * (ell[2:] + 1))

        np.random.seed(self.cmb_seed)
        alms = hp.synalm([cl_tt, cl_ee, cl_bb, cl_te], lmax=synlmax, new=True)

        beam_cut = np.ones(3 * self.nside)
        for ac in alms:
            hp.almxfl(ac, beam_cut, inplace=True)
        cmb = np.array(hp.alm2map(alms, nside=self.nside, pol=True))
        return cmb
