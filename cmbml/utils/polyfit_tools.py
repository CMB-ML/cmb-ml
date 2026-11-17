import numpy as np
from numpy.polynomial.legendre import legvander, legval
from numpy.polynomial.chebyshev import chebvander, chebval
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def setup_skl_pipeline(basis_kind, degree):
    basis = basis_kind
    degree = degree
    if basis == "monomial":
        feat = PolynomialFeatures(degree=degree, include_bias=True)
    elif basis == "legendre":
        feat = LegendreFeatures(degree=degree)
    elif basis == "chebyshev":
        feat = ChebyshevFeatures(degree=degree)
    else:
        raise ValueError(f"Unknown basis: {basis}")

    model = make_pipeline(
        feat,
        LinearRegression(fit_intercept=False),
    )
    return model


class LegendreFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, degree):
        self.degree = degree

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        x = np.asarray(X).reshape(-1)
        return legvander(x, self.degree)


class ChebyshevFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, degree):
        self.degree = degree

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        x = np.asarray(X).reshape(-1)
        return chebvander(x, self.degree)


def scale_ell(ell, out_range=(0.0, 1.0)):
    r"""
    Linearly map ell from [ell_min, ell_max] --> [a, b]

    Parameters
    ----------
    ell : array-like
        Input ell values
    out_range : tuple (a, b)
        Desired output range

    Returns
    -------
    x : ndarray, shape (n, 1)
        Rescaled values
    """
    ell = np.asarray(ell)
    a, b = out_range

    ell_min = ell.min()
    ell_max = ell.max()

    # Normalize to [0, 1]
    x = (ell - ell_min) / (ell_max - ell_min)

    # Scale to [a, b]
    x = a + (b - a) * x

    return x.reshape(-1, 1)
