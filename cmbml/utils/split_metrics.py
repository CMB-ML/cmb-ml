"""Per-multipole evaluation metrics for matched predictions and theory spectra.

SCOPE
-----
Metrics are calculated within an evaluation split, using matched predictions
and targets of shape (N, n_ell). Each prediction is compared against the
theory spectrum belonging to the same simulation.

Unlike spectrum_metrics.py, which uses a cosmological theory bank,
distributional quantities here are calculated from the evaluation split.
Consequently, R2 uses the evaluation-set theory variance rather than a
prior-based reference variance.

METRIC ORGANIZATION
-------------------
The primary evaluation metrics follow the manuscript's five criteria:

1. Predictive skill: MSE, R2, and skill relative to MeanTheory.
2. Accuracy: mean absolute relative error (MARE).
3. Bias: mean signed residual and tension.
4. Scatter: residual and prediction variance and standard deviation.
5. Sensitivity: correlation r and response amplitude A.

Additional diagnostics, including the R2 decomposition and cosmic-variance
normalizations, are retained for comparison with earlier analyses.

MeanTheory is calculated exclusively from the training-set theory spectra.
Its spectrum is supplied separately when calculating skill.

DEFINITIONS
-----------
For simulation i and multipole ell:

    residual[i, ell] = pred[i, ell] - thry[i, ell]

Variances and covariances use ddof=0, consistent with the manuscript.
Nonlinear per-instance quantities are calculated before aggregation
across simulations.

R2 is calculated directly:

    R2 = 1 - mse / thry_var

The identity

    R2 = r**2 - (A - r)**2 - b**2

is retained as a diagnostic decomposition wherever its constituent terms
are defined.

Undefined metrics are represented by NaN. Zero variance itself remains
a valid result.

UNITS
-----
The inputs must use consistent units.

Bias and standard deviations have the units of the input spectra.
MSE, variances, and cosmic variance have squared units.
Correlations, relative errors, skill, tension, and response amplitudes
are dimensionless. MARE is returned as a percentage.

No explicit unit handling is performed.
"""
import numpy as np


# --------------------------------------------------------------------------- #
# ddof
# --------------------------------------------------------------------------- #

DDOF = 0
"""Degrees-of-freedom correction for variances and standard deviations.

DDOF=0 matches the manuscript's definitions, which use the population
variance of the evaluation simulations.

It also ensures that the R2 decomposition

    R2 = r**2 - (A - r)**2 - b**2

agrees with the direct calculation

    R2 = 1 - mse / thry_var

Changing DDOF does not change MSE, but changes the target variance
used in R2 and the normalization of the decomposition terms.

Retained as a module constant for compatibility with older analyses.
The new evaluation uses DDOF=0.
"""

FSKY_EXPONENT = 1
"""Power of fsky in the cosmic-variance denominator.

DEFAULT 1, the mode-counting result::

    Var(C_ell) = 2 * C_ell**2 / ((2*ell + 1) * fsky)

EXPOSED AS A MODULE CONSTANT ONLY to reproduce the original analysis, which
applied the correction as (D_ell / fsky)**2 -- i.e. exponent 2, a further factor
of 1/sqrt(fsky) too much. At fsky = 0.7695 that made sigma_cv about 14% too
large. Set ``split_metrics.FSKY_EXPONENT = 2`` once, before computing metrics, to
reproduce it; the default is correct and should stay that way for new results.

Like DDOF, this is a global compatibility switch rather than a per-call
argument: mixing conventions within one set of metrics would be invisible.

Affects only cosmic_var and anything built on it (treble_cosmic_var, e_cv).
mse, R^2 and its decomposition, tension, rel_err, pred_var and thry_var are all
untouched by fsky at any exponent.
"""


# --------------------------------------------------------------------------- #
# shape validation
# --------------------------------------------------------------------------- #

def _as_matched(pred, thry, name="metric"):
    """Coerce to float arrays and require a matched, per-instance target.

    Matched means thry.shape == pred.shape: prediction i is compared against
    *its own* theory spectrum i. Broadcasting an ensemble-mean theory spectrum
    against every prediction is a different quantity -- it folds the theory
    ensemble's own scatter into the numerator -- and is rejected here rather
    than silently allowed, because the notebook this replaces did exactly that
    for smoothed spectra and then plotted the result alongside the matched
    version.
    """
    pred = np.asarray(pred, dtype=float)
    thry = np.asarray(thry, dtype=float)
    if pred.ndim != 2:
        raise ValueError(f"{name}: expected pred (N, n_ell), got {pred.shape}")
    if thry.shape != pred.shape:
        raise ValueError(
            f"{name}: needs matched per-instance targets; got pred {pred.shape} "
            f"vs thry {thry.shape}. Pass the per-realization theory spectra, not "
            f"their ensemble mean."
        )
    return pred, thry


def _as_ells(ells, n_ell, name="metric"):
    ells = np.asarray(ells)
    if ells.size != n_ell:
        raise ValueError(
            f"{name}: ells has {ells.size} entries but spectra have {n_ell}. "
            "Crop the spectra to the ell axis before scoring."
        )
    return ells


# --------------------------------------------------------------------------- #
# distributional quantities (split-scope, inherently across-instance)
# --------------------------------------------------------------------------- #

def thry_var(thry):
    """Across-instance variance of the theory spectra, per ell. Shape (n_ell,).

    The split-scope stand-in for the bank's S1. Named for what it is rather
    than with a Greek symbol, deliberately.

    Zero at any ell where the split has a single fixed cosmology (e.g. TestFP),
    which makes A, b and R^2 undefined there; ``decomposition`` returns NaN in
    that case rather than inf.
    """
    return np.asarray(thry, dtype=float).var(0, ddof=DDOF)


def cosmic_var(thry, ells, fsky=1.0):
    """Per-instance cosmic variance of each theory spectrum. Shape (N, n_ell).

        Var(D_ell^i) = 2 * (D_ell^i)**2 / ((2*ell + 1) * fsky)

    This is the mode-counting result: 1/fsky on the variance, 1/sqrt(fsky) on
    the standard deviation.

    CHANGED FROM THE ORIGINAL ANALYSIS, deliberately. The notebook this replaces
    applied the correction as (D_ell / fsky)**2, i.e. 1/fsky**2 on the variance
    -- a further factor of 1/sqrt(fsky) too much. At fsky = 0.7695 that made
    sigma_cv about 14% too large (1.140 vs 1.000 relative to the correct value),
    and anything normalized by it correspondingly too small.

    TO REPRODUCE THE ORIGINAL NUMBERS, set ``FSKY_EXPONENT = 2`` -- see its
    docstring at the top of the module. Do not patch this function: the constant
    is read at call time and reaches every consumer, including
    ``compute_metrics``.

    PER INSTANCE by design: aggregate with ``.mean(0)`` at the call site, which
    gives the square-then-mean ordering (see module docstring).

    fsky defaults to 1.0 (full sky, no correction). It is a scalar multiplying
    every instance and every ell identically, so it cancels out of any
    comparison at fixed fsky -- including the degree-selection criterion. It
    affects only the cosmic-variance-normalized outputs: ``cosmic_var`` itself,
    ``treble_cosmic_var``, and any e_cv built on them.

    NOTE ON THE fsky VALUE ITSELF, which is a separate question from the
    exponent. The original estimates it as sum(w**2)/n_pix on a Gaussian-
    smoothed mask (see ``data_helper.compute_fsky``), a w2-style estimator.
    Apodization makes the "right" fsky genuinely ambiguous, so that is a
    defensible convention -- but the notebook's own markdown records a lack of
    confidence in it. The exponent above is not a convention question; the value
    passed in still is.
    """
    thry = np.asarray(thry, dtype=float)
    ells = _as_ells(ells, thry.shape[-1], "cosmic_var")
    nu = 2.0 * ells + 1.0
    return 2.0 * thry ** 2 / (nu * fsky ** FSKY_EXPONENT)


# --------------------------------------------------------------------------- #
# per-instance metrics (shape (N, n_ell); caller aggregates)
# --------------------------------------------------------------------------- #

def squared_error(pred, thry):
    """Per-instance squared error. Shape (N, n_ell). ``.mean(0)`` gives MSE."""
    pred, thry = _as_matched(pred, thry, "squared_error")
    return (pred - thry) ** 2


def rel_err(pred, thry):
    """Per-instance signed fractional error (pred - thry) / thry. Shape (N, n_ell).

    Signed and fractional. The notebook plotted mean(|.|) * 100; that is
    ``np.abs(rel_err(...)).mean(0) * 100``, which the caller composes. Keeping
    the sign here means the same primitive serves both, and a signed mean
    distinguishes a biased model from a merely noisy one.

    Entries where thry == 0 are NaN; aggregate with ``np.nanmean``.
    """
    pred, thry = _as_matched(pred, thry, "rel_err")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(
            pred - thry, thry,
            out=np.full_like(pred, np.nan),
            where=(thry != 0),
        )


def treble(pred, thry, denom):
    """RMS residual in units of a chosen spread, per ell. Shape (n_ell,).

        treble(ell) = sqrt( (1/N) * sum_i (pred_i - thry_i)**2 / denom )

    Matched per-instance targets, per the definition: the superscript i sits on
    both the prediction and the truth. The alternative spelling found in the
    notebook -- broadcasting the ensemble-mean theory spectrum -- measures
    distribution adherence rather than prediction quality and is rejected by
    ``_as_matched``.

    ``denom`` is variance-like and may be per-instance (N, n_ell), e.g. the
    output of ``cosmic_var``, or aggregate (n_ell,), e.g. ``thry_var`` or the
    prediction variance. Both broadcast correctly.

    N is taken from ``pred``. The original took it from a module-level constant,
    which silently gave the wrong normalization on any array that did not happen
    to have exactly that many rows.
    """
    pred, thry = _as_matched(pred, thry, "treble")
    denom = np.asarray(denom, dtype=float)
    n = pred.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(np.sum((pred - thry) ** 2 / denom, axis=0) / n)


# --------------------------------------------------------------------------- #
# aggregate metrics (shape (n_ell,); no per-instance value exists)
# --------------------------------------------------------------------------- #

def _has_spread(sd_t, mu_t, rtol=1e-10):
    """Whether the target genuinely varies across instances, per ell.

    Compares the spread against the magnitude of the values rather than against
    exactly zero. A fixed-cosmology split has identical rows, but ``std(0)`` of
    identical floats is not exactly 0 -- numpy's variance leaves residue around
    1e-11 on values of order 1e3. A ``sd_t > 0`` test therefore passes, and A and
    b are then divided by that residue, producing R^2 around -1e27 instead of
    NaN.

    The margin is wide in both directions: residue sits ~1e-14 relative, while a
    real cosmological spread is ~1e-2 relative, so rtol=1e-10 separates them by
    eight orders either way.
    """
    return sd_t > rtol * np.abs(mu_t)


def decomposition(pred, thry, ells):
    """Calculate R2 and its diagnostic decomposition, per multipole.

    R2 is calculated directly from matched prediction errors:

        R2 = 1 - mse / thry_var

    This remains defined for constant predictions when the targets vary.

    The decomposition expresses R2 as:

        R2 = r**2 - (A - r)**2 - b**2

    where:

        r              correlation between predictions and targets
        A              prediction std / target std
        b              mean residual / target std
        ceiling_r2     r**2
        loss_calib     (A - r)**2
        loss_bias      b**2

    The decomposition separates tracking, calibration, and bias losses.
    Its terms are diagnostics rather than the method used to calculate R2.

    The identity holds when the constituent terms are defined and all
    quantities use the same evaluation-set target variance with ddof=0.

    Undefined cases
    ---------------
    If the target theory has zero variance, R2, A, b, and r are undefined.

    If predictions are constant but targets vary, R2 remains defined,
    A is zero, and r is undefined. Consequently, the complete
    decomposition cannot be evaluated.

    Returns
    -------
    dict
        Per-multipole arrays containing R2 and decomposition quantities.
        Undefined quantities are represented by NaN.
    """
    pred, thry = _as_matched(pred, thry, "decomposition")
    ells = _as_ells(ells, pred.shape[1], "decomposition")

    mu_p, mu_t = pred.mean(0), thry.mean(0)
    sd_p, sd_t = pred.std(0, ddof=DDOF), thry.std(0, ddof=DDOF)

    mse = squared_error(pred, thry).mean(0)

    with np.errstate(divide="ignore", invalid="ignore"):
        varies = _has_spread(sd_t, mu_t)
        pred_varies = _has_spread(sd_p, mu_p)
        A = np.where(
            varies,
            np.where(pred_varies, sd_p / sd_t, 0.0),
            np.nan,
        )
        b = np.where(varies, (mu_p - mu_t) / sd_t, np.nan)
        cov = ((pred - mu_p) * (thry - mu_t)).sum(0) / (pred.shape[0] - DDOF)
        r = np.where(
            varies & pred_varies,
            cov / (sd_p * sd_t),
            np.nan,
        )
        R2 = np.divide(
            -mse,
            sd_t**2,
            out=np.full_like(mse, np.nan),
            where=varies,
        )
        R2 = np.where(varies, 1.0 + R2, np.nan)

    ceiling_r2 = r ** 2
    loss_calib = (A - r) ** 2
    loss_bias = b ** 2

    return {
        "ell": ells,
        # predictive skill
        "R2": R2,
        # sensitivity
        "A": A,
        "r": r,
        "b": b,
        "A_minus_r": A - r,
        "ceiling_r2": ceiling_r2,
        "loss_calib": loss_calib,
        "loss_bias": loss_bias,
    }


def tension(pred, thry):
    """|<E>| / sigma(E) per ell, where E = pred - thry. Shape (n_ell,).

    The mean error in units of its own across-instance spread: ~1 means the
    offset is the size of the scatter, <<1 means it is buried in it, >>1 means a
    bias large relative to residual scatter.

    Algebraically identical to the notebook's

        abs_err / sqrt(var_pred + var_thry - 2*cov)

    since sigma(E)**2 expands to exactly that. The notebook carried both
    spellings and reconciled them in a later cell; there is one here.
    """
    pred, thry = _as_matched(pred, thry, "tension")

    residual = pred - thry
    me = residual.mean(axis=0)
    sd_err = residual.std(axis=0, ddof=DDOF)

    varies = _has_spread(sd_err, me)

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(
            np.abs(me),
            sd_err,
            out=np.full_like(me, np.nan),
            where=varies,
        )


def e_thry(pred, thry):
    """Signed mean error in units of the theory spread. Shape (n_ell,).

    <E> / sqrt(thry_var). Split-scope replacement for the bank-normalized
    e_thry; same quantity, local normalizer.
    """
    pred, thry = _as_matched(pred, thry, "e_thry")
    me = pred.mean(0) - thry.mean(0)
    sd_t = thry.std(0, ddof=DDOF)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(_has_spread(sd_t, thry.mean(0)), me / sd_t, np.nan)


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #

def compute_metrics(pred, thry, ells, fsky=1.0, mean_theory=None):
    """Calculate per-multipole evaluation metrics for one model and split.

    Predictions and targets must be matched arrays with shape
    (N, n_ell). All returned metrics have shape (n_ell,).

    Metrics follow the manuscript's five evaluation criteria:
    predictive skill, accuracy, bias, scatter, and sensitivity.

    Parameters
    ----------
    pred : array_like
        Predicted power spectra, shape (N, n_ell).
    thry : array_like
        Matched theory power spectra, shape (N, n_ell).
    ells : array_like
        Multipole values corresponding to the final array dimension.
    fsky : float, optional
        Effective sky fraction for cosmic-variance calculations.
        Does not affect the primary evaluation metrics.
    mean_theory : array_like, optional
        Training-set mean theory spectrum, shape (n_ell,).
        Required to calculate skill relative to MeanTheory.

    Returns
    -------
    dict[str, ndarray]
        Per-multipole evaluation metrics and supporting diagnostics.

        Undefined quantities are represented by NaN. If mean_theory
        is omitted, skill and baseline_mse are returned as NaN arrays.

    Notes
    -----
    Skill compares model MSE against the MSE obtained by predicting
    the training-set mean theory spectrum for every simulation.

    Scalar summaries over multipole ranges are not calculated here.
    In particular, scalar skill must be calculated from aggregated
    MSE values rather than by averaging per-multipole skill.
    """
    pred, thry = _as_matched(pred, thry, "compute_metrics")
    ells = _as_ells(ells, pred.shape[1], "compute_metrics")

    # Shared calculations
    residual = pred - thry

    bias = residual.mean(axis=0)
    resid_var = residual.var(axis=0, ddof=DDOF)
    resid_std = np.sqrt(resid_var)

    pred_var = pred.var(axis=0, ddof=DDOF)
    pred_std = np.sqrt(pred_var)

    mse = (residual**2).mean(axis=0)

    cv = cosmic_var(thry, ells, fsky=fsky)
    tv = thry_var(thry)

    decomp = decomposition(pred, thry, ells)

    # MeanTheory reference
    if mean_theory is not None:
        mean_theory = np.asarray(mean_theory, dtype=float)

        if mean_theory.shape != (pred.shape[1],):
            raise ValueError(
                f"MeanTheory must have shape ({pred.shape[1]},), "
                f"got {mean_theory.shape}."
            )

        baseline_mse = (
            (thry - mean_theory[np.newaxis, :])**2
        ).mean(axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            skill = 1.0 - np.divide(
                mse,
                baseline_mse,
                out=np.full_like(mse, np.nan),
                where=baseline_mse > 0,
            )
    else:
        baseline_mse = np.full_like(mse, np.nan)
        skill = np.full_like(mse, np.nan)

    return {
        "ell": ells,

        # 1. Predictive skill
        "mse": mse,
        "R2": decomp["R2"],
        "skill": skill,

        # 2. Accuracy
        "abs_rel_err_pct": np.nanmean(
            np.abs(rel_err(pred, thry)), axis=0
        ) * 100.0,

        # 3. Bias
        "bias": bias,
        "tension": tension(pred, thry),

        # 4. Scatter
        "resid_var": resid_var,
        "resid_std": resid_std,
        "pred_var": pred_var,
        "pred_std": pred_std,

        # 5. Sensitivity
        "r": decomp["r"],
        "A": decomp["A"],

        # Supporting quantities
        "thry_var": tv,
        "baseline_mse": baseline_mse,
        "rel_err": np.nanmean(rel_err(pred, thry), axis=0),
        "cosmic_var": cv,
        "e_thry": e_thry(pred, thry),

        # R2 decomposition diagnostics
        "b": decomp["b"],
        "A_minus_r": decomp["A_minus_r"],
        "ceiling_r2": decomp["ceiling_r2"],
        "loss_calib": decomp["loss_calib"],
        "loss_bias": decomp["loss_bias"],

        # Legacy treble diagnostics
        "treble_cosmic_var": treble(pred, thry, cv),
        "treble_thry_var": treble(
                pred,
                thry,
                np.where(
                    _has_spread(np.sqrt(tv), thry.mean(axis=0)),
                    tv,
                    np.nan,
                ),
            ),
        "treble_pred_var": treble(pred, thry, pred_var),
    }
