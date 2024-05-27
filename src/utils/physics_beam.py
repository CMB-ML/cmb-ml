import healpy as hp
from astropy.io import fits

import numpy as np


class Beam:
    def __init__(self, beam) -> None:
        self.beam = beam

    def conv1(self, ps):
        ps_applied = ps * self.beam
        return ps_applied

    def conv2(self, ps):
        ps_applied = ps * (self.beam ** 2)
        return ps_applied

    def deconv1(self, ps):
        # TODO: Handle zeros in beam
        ps_applied = ps / self.beam

        # TODO: Use this method instead
        # log_ps = np.log(ps)
        # log_beam = np.log(self.beam)
        # log_applied = log_ps - log_beam
        # ps_applied = np.exp(log_applied)
        return ps_applied

    def deconv2(self, ps):
        # TODO: Handle zeros in beam
        ps_applied = ps / (self.beam ** 2)

        # TODO: Use this method instead
        # log_ps = np.log(ps)
        # log_beam = np.log(self.beam)
        # log_applied = log_ps - 2 * log_beam
        # ps_applied = np.exp(log_applied)
        return ps_applied


class GaussianBeam(Beam):
    def __init__(self, beam_fwhm, lmax) -> None:
        """
        beam_fwhm in arcmin
        """
        # Convert fwhm from arcmin to radians
        self.fwhm = beam_fwhm * np.pi / (180*60)
        self.lmax = lmax
        beam = hp.gauss_beam(self.fwhm, lmax=lmax)
        super().__init__(beam)


class PlanckBeam(Beam):
    def __init__(self, planck_path, lmax) -> None:
        self.planck_path = planck_path
        self.lmax = lmax
        beam = get_planck_beam(planck_path, lmax)
        super().__init__(beam)


class NoBeam(Beam):
    def __init__(self, lmax) -> None:
        self.lmax = lmax
        beam = np.ones(lmax)
        super().__init__(beam)


def get_planck_beam(planck_path, lmax):
    hdul = fits.open(planck_path)
    beam = hdul[2].data['INT_BEAM']
    return Beam(beam[:lmax+1])
