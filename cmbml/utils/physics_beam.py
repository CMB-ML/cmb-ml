import healpy as hp
from astropy.io import fits
import pysm3.units as u

import numpy as np


class Beam:
    """
    Utility class representing an detector beam for convolution or deconvolution.
    Beams may need to be squared for autopower spectra.

    Attributes:
        beam (np.ndarray): The beam profile used for convolution or deconvolution.

    Methods:
        conv1(ps): Apply the beam to the input power spectrum using convolution.
        conv2(ps): Apply the squared beam to the input power spectrum using convolution.
        deconv1(ps): Remove the effect of the beam from the input power spectrum using deconvolution.
        deconv2(ps): Remove the effect of the squared beam from the input power spectrum using deconvolution.
    """

    def __init__(self, beam) -> None:
        arr = np.asarray(beam, dtype=float)
        if arr.ndim != 1:
            raise ValueError(f"Beam must be 1D, got shape {arr.shape}")
        self.beam = arr

    def conv1(self, ps):
        """
        Apply the beam to the input power crosspectrum using convolution.
        The expectation is that this beam is for one map and another beam will
        be applied for the other map.

        Args:
            ps (np.ndarray): The input power spectrum.

        Returns:
            np.ndarray: The power spectrum with the beam applied.
        """
        ps_applied = ps * self.beam
        return ps_applied

    def conv2(self, ps):
        """
        Apply the squared beam to the input power autospectrum using convolution.
        This beam is effectively applied twice (for the same map appearing twice in the
        autopower spectrum calculation)

        Args:
            ps (np.ndarray): The input power spectrum.

        Returns:
            np.ndarray: The power spectrum with the squared beam applied.
        """
        ps_applied = ps * (self.beam ** 2)
        return ps_applied

    def deconv1(self, ps):
        """
        Remove the effect of the beam from the input power spectrum using deconvolution.
        The expectation is that this beam is for one map and another beam will
        be applied for the other map.

        Args:
            ps (np.ndarray): The input power spectrum.

        Returns:
            np.ndarray: The power spectrum with the beam effect removed.
        """
        # TODO: Handle zeros in beam
        ps_applied = ps / self.beam

        # TODO: Use this method instead
        # log_ps = np.log(ps)
        # log_beam = np.log(self.beam)
        # log_applied = log_ps - log_beam
        # ps_applied = np.exp(log_applied)
        return ps_applied

    def deconv2(self, ps):
        """
        Remove the effect of the squared beam from the input power spectrum using deconvolution.
        This beam is effectively applied twice (for the same map appearing twice in the
        autopower spectrum calculation)

        Args:
            ps (np.ndarray): The input power spectrum.

        Returns:
            np.ndarray: The power spectrum with the squared beam effect removed.
        """
        # TODO: Handle zeros in beam
        ps_applied = ps / (self.beam ** 2)

        # TODO: Use this method instead
        # log_ps = np.log(ps)
        # log_beam = np.log(self.beam)
        # log_applied = log_ps - 2 * log_beam
        # ps_applied = np.exp(log_applied)
        return ps_applied


class GaussianBeam(Beam):
    """
    Utility class inheriting from the Beam class and representing a Gaussian beam.

    Attributes:
        beam_fwhm (float): The full width at half maximum of the beam in arcmin.
        lmax (int): The maximum multipole moment of the beam.
    """
    def __init__(self, beam_fwhm, lmax) -> None:
        # Convert fwhm from arcmin to radians
        try:
            self.fwhm = beam_fwhm.to(u.rad).value
        except:
            self.fwhm = beam_fwhm * np.pi / (180*60)
        self.lmax = lmax
        beam = hp.gauss_beam(self.fwhm, lmax=lmax)
        super().__init__(beam)


class PlanckBeam(Beam):
    """
    Utility class inheriting from the Beam class and representing a Planck beam.

    Attributes:
        planck_path (str): The path to the Planck data.
        lmax (int): The maximum multipole moment of the beam.
    """
    def __init__(self, planck_path, lmax) -> None:
        self.planck_path = planck_path
        self.lmax = lmax
        beam = get_planck_beam(planck_path, lmax)
        super().__init__(beam)


class NoBeam(Beam):
    """
    Utility class inheriting from the Beam class, representing no beam, for use
    with raw simulated signals.

    Attributes:
        lmax (int): The maximum multipole moment of the beam.
    """
    def __init__(self, lmax) -> None:
        self.lmax = lmax
        beam = np.ones(lmax+1)
        super().__init__(beam)


def get_planck_beam(planck_path, lmax):
    """
    Retrieve the Planck beam from Planck data given an lmax.

    Args:
        planck_path (str): The path to the Planck data.
        lmax (int): The maximum multipole moment of the beam.

    Returns:
        np.ndarray: The Planck beam.
    """
    hdul = fits.open(planck_path)
    beam = hdul[2].data['INT_BEAM']
    return Beam(beam[:lmax+1])


def ensure_beam(obj, *, lmax=None, on_none="identity"):
    """
    Accepts Beam | array-like | None → returns Beam.
    - If obj is Beam, return as-is.
    - If obj is None and on_none == 'identity', require lmax and return NoBeam(lmax).
    - Else, coerce to array and wrap as Beam.
    """
    if isinstance(obj, Beam):
        return obj
    if obj is None:
        if on_none == "identity":
            if lmax is None:
                raise ValueError("lmax required to construct NoBeam when beam=None")
            return NoBeam(lmax)
        raise ValueError("beam=None not allowed here")
    arr = np.asarray(obj, dtype=float)
    if lmax is not None and arr.shape[0] < lmax + 1:
        raise ValueError(f"Beam array too short for lmax={lmax}: len={arr.shape[0]}")
    return Beam(arr if lmax is None else arr[:lmax+1])
