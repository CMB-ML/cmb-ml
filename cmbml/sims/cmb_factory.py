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
    def __init__(
        self,
        nside,
        cmb_spectra,
        max_nside=None,
        cmb_seed=None,
        map_dist=None
        ):
        """Lensed CMB

        Takes an input unlensed CMB power spectrum from CAMB and uses
        part of the Taylens code and synfast to generate correlated CMB maps.
        Code tested with power spectra including lensing potential.

        Parameters
        ----------

        cmb_spectra : path
            Input text file from CAMB, spectra unlensed
        cmb_seed : int
            Numpy random seed for synfast, set to None for a random seed
        apply_delens : bool
            If true, simulate delensing with taylens
        delensing_ells : path
            Space delimited file with ells in the first columns and suppression
            factor (1 for no suppression) in the second column
        """
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
