"""Interchangeable polynomial smoothers for power spectra.

A smoother maps (ell, D_ell) -> smoothed D_ell on the same ell grid, at a fixed
degree. Implementations are interchangeable by construction so that the degree
selection in ``degree_selection.py`` can be run against any of them and the
results compared.

Two families are provided:

  ``polyfit``    numpy's np.polyfit on the monomial basis, with ell rescaled to
                 [-1, 1]. This reproduces the original analysis exactly and is
                 the default for that reason.

  ``legendre``   least squares on an orthogonal basis via scikit-learn, from
  ``chebyshev``  ``polyfit_tools``. Mathematically the same fit as the monomial
  ``monomial``   version -- same function space, same minimizer -- but far
                 better conditioned at the degrees in use here.

CONDITIONING, and why the basis choice is not cosmetic. A degree-25 monomial
Vandermonde on [-1, 1] has a condition number around 1e9-1e12; the orthogonal
equivalents stay near 1e2. The original sweep ran to degree 30 on monomials,
inside a ProcessPoolExecutor where numpy's RankWarning is raised in the worker
process and never surfaces in the notebook. So the high-degree end of that sweep
may carry numerical noise on top of whatever the statistics are doing.
``PolyfitSmoother`` records rank warnings (see ``last_fit_ill_conditioned``) so
this is observable rather than assumed.

All smoother classes are top-level with plain attributes, so instances pickle
and can cross a ProcessPoolExecutor boundary.
"""

import sys
import warnings
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, "/home/jim/VSCode_Projects_SPACE/e-d2ps/")
from cmbml.utils.polyfit_tools import setup_skl_pipeline, scale_ell

# numpy moved RankWarning between 1.x and 2.x
_RANK_WARNING = getattr(np, "RankWarning", None) or np.exceptions.RankWarning


# --------------------------------------------------------------------------- #
# smoothers
# --------------------------------------------------------------------------- #

@dataclass
class PolyfitSmoother:
    """np.polyfit on monomials, ell rescaled to [-1, 1]. The original method.

    Reproduces the notebook's ``polynomial_fit`` exactly: same rescaling, same
    solver, same basis. Kept as the default so degree selections computed here
    match the ones already in circulation.

    Ill-conditioned fits set ``last_fit_ill_conditioned``. This is per-instance
    state, not part of the fit, so it is excluded from equality and repr; read
    it immediately after a call, or use ``smooth_ensemble`` which tallies it.
    """

    degree: int
    ill_conditioned_count: int = field(default=0, compare=False, repr=False)

    @property
    def name(self):
        return "polyfit"

    def smooth(self, ell, D_ell):
        ell = np.asarray(ell, dtype=float)
        D_ell = np.asarray(D_ell, dtype=float)
        u = _to_unit_interval(ell, (-1.0, 1.0))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", _RANK_WARNING)
            coeffs = np.polyfit(u, D_ell, self.degree)
        self.last_fit_ill_conditioned = any(
            issubclass(w.category, _RANK_WARNING) for w in caught
        )
        if self.last_fit_ill_conditioned:
            self.ill_conditioned_count += 1
        return np.polyval(coeffs, u)


@dataclass
class BasisSmoother:
    """Least squares on an orthogonal (or monomial) basis, via polyfit_tools.

    ``basis`` is one of 'legendre', 'chebyshev', 'monomial'.

    RANGE. Legendre and Chebyshev polynomials are orthogonal on [-1, 1], and the
    conditioning benefit that motivates using them is only realised there.
    ``polyfit_tools.scale_ell`` defaults to (0, 1), so the range is passed
    explicitly below rather than relying on that default -- a Legendre fit on
    [0, 1] is legal but throws away the reason to have chosen Legendre. The
    monomial case is put on [-1, 1] too, for parity with PolyfitSmoother.
    """

    degree: int
    basis: str = "legendre"
    out_range: tuple = (-1.0, 1.0)

    @property
    def name(self):
        return self.basis

    def smooth(self, ell, D_ell):
        D_ell = np.asarray(D_ell, dtype=float)
        x = scale_ell(ell, out_range=self.out_range)     # (n, 1) for sklearn
        model = setup_skl_pipeline(self.basis, self.degree)
        model.fit(x, D_ell)
        return np.asarray(model.predict(x), dtype=float)


SMOOTHERS = {
    "polyfit": lambda degree, **kw: PolyfitSmoother(degree=degree, **kw),
    "legendre": lambda degree, **kw: BasisSmoother(degree=degree, basis="legendre", **kw),
    "chebyshev": lambda degree, **kw: BasisSmoother(degree=degree, basis="chebyshev", **kw),
    "monomial": lambda degree, **kw: BasisSmoother(degree=degree, basis="monomial", **kw),
}


def make_smoother(kind, degree, **kwargs):
    """Construct a smoother by name. See ``SMOOTHERS`` for the registry."""
    try:
        factory = SMOOTHERS[kind]
    except KeyError:
        raise ValueError(
            f"unknown smoother {kind!r}; choose from {sorted(SMOOTHERS)}"
        ) from None
    return factory(degree, **kwargs)


# --------------------------------------------------------------------------- #
# ensemble application
# --------------------------------------------------------------------------- #

def smooth_ensemble(smoother, ell, D_ells, n_workers=None):
    """Apply a smoother to every row of ``D_ells``. Returns (N, n_ell).

    Serial by default. Pass ``n_workers`` to fan out over processes; each worker
    gets its own smoother copy, so any per-instance state (the rank-warning
    tally) is aggregated back onto the passed smoother at the end rather than
    being lost in the child processes -- which is how the original analysis came
    to run 27 degrees of monomial fits without ever seeing a conditioning
    warning.
    """
    D_ells = np.asarray(D_ells, dtype=float)
    if D_ells.ndim != 2:
        raise ValueError(f"expected D_ells (N, n_ell), got {D_ells.shape}")
    ell = np.asarray(ell, dtype=float)

    if n_workers is None or n_workers == 1:
        return np.stack([smoother.smooth(ell, row) for row in D_ells])

    from concurrent.futures import ProcessPoolExecutor

    tasks = [(smoother, ell, row) for row in D_ells]
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(_smooth_one, tasks))

    out = np.stack([r[0] for r in results])
    if hasattr(smoother, "ill_conditioned_count"):
        smoother.ill_conditioned_count += sum(r[1] for r in results)
    return out


def _smooth_one(args):
    """Module-level worker so it pickles for ProcessPoolExecutor."""
    smoother, ell, row = args
    y = smoother.smooth(ell, row)
    return y, int(getattr(smoother, "last_fit_ill_conditioned", False))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _to_unit_interval(ell, out_range=(-1.0, 1.0)):
    """Linearly map ell onto ``out_range``. 1-D, unlike polyfit_tools.scale_ell,
    which reshapes to (n, 1) for sklearn's benefit."""
    ell = np.asarray(ell, dtype=float)
    a, b = out_range
    lo, hi = ell.min(), ell.max()
    if hi == lo:
        raise ValueError("ell has zero range; cannot rescale")
    return a + (b - a) * (ell - lo) / (hi - lo)
