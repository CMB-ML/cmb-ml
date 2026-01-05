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
