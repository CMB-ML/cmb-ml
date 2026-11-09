from dataclasses import dataclass, field, replace, asdict
from pathlib import Path
import numpy as np
import camb

from cmbml.core.asset_handlers import CambPowerSpectrum


@dataclass  # CAMBSettings Dataclass definition. Create CS_512
class CAMBSettings:
    # Cosmological parameters
    ombh2: float = 0.02264
    omch2: float = 0.1138
    tau: float = 0.089
    As: float = 2.41e-9
    ns: float = 0.972
    cosmomc_theta: float = 0.0104150
    pivot_scalar: float = 0.002

    # CAMB calculation
    lmax: int = 1280
    lens_potential_accuracy: int = 0

    # Spectrum output
    CMB_unit: str = "muK"


def make_camb_params(  # Interface between CAMBSettings and actual py CAMB library
    settings: CAMBSettings,
    do_lensing: bool,
):
    pars = camb.CAMBparams()

    pars.set_cosmology(
        ombh2=settings.ombh2,
        omch2=settings.omch2,
        tau=settings.tau,
        cosmomc_theta=settings.cosmomc_theta,
    )

    pars.InitPower.set_params(
        As=settings.As,
        ns=settings.ns,
        pivot_scalar=settings.pivot_scalar,
    )

    pars.set_for_lmax(
        settings.lmax,
        lens_potential_accuracy=settings.lens_potential_accuracy,
    )

    pars.DoLensing = bool(do_lensing)
    return pars


data = camb.get_results(make_camb_params(settings=CAMBSettings(), do_lensing=True))

test_path = Path("test_camb_unlensed.dat")
lmax = 1280

handler = CambPowerSpectrum()
handler.write(test_path, data, lmax)

# Read the saved file
table = np.loadtxt(test_path)

# Retrieve the expected spectra directly from CAMB
unlensed = data.get_unlensed_scalar_cls(lmax=lmax, CMB_unit="muK")
lensing = data.get_lens_potential_cls(lmax=lmax, CMB_unit="muK")
lensed = data.get_total_cls(lmax=lmax, CMB_unit="muK")

# Check multipoles and column ordering
np.testing.assert_array_equal(table[:, 0], np.arange(lmax + 1))

# Check unlensed TT, EE, BB, TE
np.testing.assert_allclose(
    table[:, 1:5], unlensed[:, :4], rtol=1e-7, atol=0
)

# Check lensing potential PP, PT, PE
np.testing.assert_allclose(
    table[:, 5:8], lensing[:, :3], rtol=1e-7, atol=0
)

# Confirm that lensing information is present
assert np.any(table[:, 5] > 0)

# Confirm that the saved TT is NOT the lensed TT
assert not np.allclose(table[:, 1], lensed[:, 0], rtol=1e-4)

print("PASS: Unlensed CMB spectra with lensing potential.")
