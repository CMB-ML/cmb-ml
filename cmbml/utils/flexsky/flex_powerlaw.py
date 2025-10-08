# This is a reimplementation of the pysm3.PowerLawRealization object.
#   It enables the user to replace components more easily, instead of 
#   recreating a Sky object and all the attendant baggage (run-time).
# This is simply the last line of the __init__ method, extracted, 
#   as a stand-alone function.


from pysm3.models import PowerLawRealization
import warnings
from contextlib import contextmanager


@contextmanager
def suppress_complex_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message=".*Casting complex values to real discards the imaginary part.*",
        )
        yield


class FlexPowerLaw(PowerLawRealization):
    def replace_draw(self, 
                     seeds,
                     synalm_lmax:int = None):
        with suppress_complex_warning():
            (
                self.I_ref,
                self.Q_ref,
                self.U_ref,
                self.pl_index,
            ) = self.draw_realization(synalm_lmax, seeds)
