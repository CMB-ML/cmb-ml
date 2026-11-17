"""Selection of a per-model polynomial smoothing degree, in two stages.

STAGE 1 -- SWEEP. ``sweep_smoothing`` smooths every prediction at every
candidate degree and keeps the results in a ``SmoothingSweep``. This is the
expensive step: one polynomial fit per (degree, realization).

STAGE 2 -- SCORE. ``score_sweep`` reduces a sweep to a degree -> score curve
under some criterion, and ``select_degree`` picks a degree from that curve.
Both are cheap, so a criterion can be swapped, re-parameterized, or compared
against others without re-smoothing anything.

The split exists because the criterion is the unsettled part. Use
``compare_criteria`` to see several at once on one cached sweep.

    sweep = sweep_smoothing(ells, pred, thry, range(4, 31))
    print(format_comparison(compare_criteria(sweep)))   # what do the options say?
    curve = score_sweep(sweep, "mean_r2")               # the one in use
    degree = select_degree(curve, "mean_r2")

THE CRITERION IN USE is ``mean_r2``: the per-ell R^2 decomposition averaged over
ell, maximized over degree. That is the collaborator's method, reproduced
faithfully including the argmax, and it remains the default.

READ THE CURVE, NOT JUST THE ARGMAX. On the data this was built for the curve is
flat from roughly M=14 to M=30, so the argmax selects among values differing by
less than their own noise. ``select_degree(..., rule="plateau")`` gives the
parsimonious alternative -- smallest degree within a tolerance of the best --
without imposing it. Report the curve alongside any chosen degree.

ELL MASKING. ``ell_max`` restricts the *scoring* range, not the fit. Spectra are
smoothed over the full range in stage 1, and stage 2 masks columns before
scoring, so the smoothed spectra stay defined everywhere for downstream figures.
Since the per-ell metrics are computed independently per ell, masking before or
after scoring is equivalent. This exists for models whose predictions degrade at
high ell for reasons a polynomial cannot fix; scoring them over the full range
would select a degree chasing that defect.

No pipeline coupling, and no knowledge of the data_helper containers: everything
here takes plain (ell, array) inputs.

SCOPE. This module serves two callers. The pipeline uses sweep_smoothing,
score_sweep and select_degree for a single model, and persists only the chosen
degree. The development notebook additionally uses compare_criteria, the plot
functions, and SmoothingSweep's to_dict/from_dict, which exist so an expensive
sweep survives a kernel restart while criteria are being compared. Nothing in
the pipeline path depends on those. (Consider separating these responsibilities
in the future.)
"""
from dataclasses import dataclass, field, asdict

import numpy as np

try:                                                     # package or flat layout
    from .smoothing import make_smoother, smooth_ensemble
    from . import split_metrics as sm
except ImportError:                                      # pragma: no cover
    from smoothing import make_smoother, smooth_ensemble
    import split_metrics as sm


# --------------------------------------------------------------------------- #
# stage 1: the sweep
# --------------------------------------------------------------------------- #

@dataclass
class SmoothingSweep:
    """Smoothed spectra at every candidate degree, for one model and split.

    The expensive artifact. Hold onto it while experimenting with criteria;
    ``save``/``load`` persist it so a kernel restart does not cost a re-sweep.

    Memory is roughly ``n_degrees * N * n_ell * 8`` bytes -- about 55 MB for 27
    degrees at 200 x 1279, so a handful of models fit comfortably in RAM.
    """

    model: str
    ells: np.ndarray
    thry: np.ndarray
    smoothed: dict                      # degree -> (N, n_ell)
    smoother: str = "polyfit"
    split: str = ""
    ill_conditioned: int = 0

    @property
    def degrees(self):
        return sorted(self.smoothed)

    def at(self, degree):
        if degree not in self.smoothed:
            raise KeyError(f"degree {degree} not in sweep; swept {self.degrees}")
        return self.smoothed[degree]

    def to_dict(self):
        """A dict of arrays, for ``np.savez_compressed(path, **d)``. No file I/O
        -- the Asset's Handler owns that.

        Metadata is stored as its own entries rather than a JSON blob, so the
        archive is inspectable and this module needs no json import.
        """
        return {
            "ells": self.ells,
            "thry": self.thry,
            "model": np.array(self.model),
            "smoother": np.array(self.smoother),
            "split": np.array(self.split),
            "ill_conditioned": np.array(self.ill_conditioned),
            "degrees": np.array(self.degrees),
            **{f"d{d}": a for d, a in self.smoothed.items()},
        }

    @classmethod
    def from_dict(cls, z):
        """Rebuild from ``to_dict`` output, or from an open ``NpzFile`` -- both
        support key lookup, so neither needs converting first."""
        degrees = [int(d) for d in z["degrees"]]
        return cls(
            model=str(z["model"]),
            ells=z["ells"],
            thry=z["thry"],
            smoothed={d: z[f"d{d}"] for d in degrees},
            smoother=str(z["smoother"]),
            split=str(z["split"]),
            ill_conditioned=int(z["ill_conditioned"]),
        )

    def __repr__(self):
        return (
            f"SmoothingSweep({self.model}/{self.split}, {self.smoother}, "
            f"degrees {min(self.degrees)}..{max(self.degrees)}, "
            f"{self.thry.shape[0]}x{self.ells.size})"
        )


def sweep_smoothing(
    ells, pred, thry, degrees, smoother="polyfit", model="", split="",
    n_workers=None, smoother_kwargs=None,
):
    """Stage 1. Smooth ``pred`` at every candidate degree. Returns a sweep.

    ``pred`` and ``thry`` are matched (N, n_ell) arrays. Smoothing is done over
    the full ell range; any restriction belongs in stage 2.
    """
    ells = np.asarray(ells, dtype=float)
    pred = np.asarray(pred, dtype=float)
    thry = np.asarray(thry, dtype=float)

    out, n_ill = {}, 0
    for degree in degrees:
        sm_obj = make_smoother(smoother, degree, **(smoother_kwargs or {}))
        out[int(degree)] = smooth_ensemble(sm_obj, ells, pred, n_workers=n_workers)
        n_ill += getattr(sm_obj, "ill_conditioned_count", 0)

    return SmoothingSweep(
        model=model, ells=ells, thry=thry, smoothed=out,
        smoother=smoother, split=split, ill_conditioned=n_ill,
    )


def sweep_from(spectra, degrees, smoother="polyfit", n_workers=None,
               smoother_kwargs=None):
    """Stage 1 from a spectra container, rather than loose arrays.

    Accepts anything exposing ``.ells``, ``.matched()``, and optionally
    ``.model`` / ``.split`` -- which ``data_helper.ModelSpectra`` does. Nothing
    is imported: this is attribute access, so the one-directional dependency
    holds and this module still knows nothing about those containers.

        sweeps = {m: sweep_from(lib.get(m, "Valid"), range(10, 31))
                  for m in SMOOTHED_MODELS}
    """
    pred, thry = spectra.matched()
    return sweep_smoothing(
        spectra.ells, pred, thry, degrees,
        smoother=smoother,
        model=getattr(spectra, "model", ""),
        split=getattr(spectra, "split", ""),
        n_workers=n_workers, smoother_kwargs=smoother_kwargs,
    )


# --------------------------------------------------------------------------- #
# stage 2: criteria
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Criterion:
    """A per-ell metric reduced to one scalar, plus which way is better.

    ``fn(pred, thry, ells) -> float`` receives arrays already masked to the
    scoring range. ``direction`` is 'max' or 'min' and is what lets
    ``select_degree`` work without being told, so a criterion measuring error
    cannot be accidentally maximized.
    """

    name: str
    fn: object
    direction: str = "max"

    def __call__(self, pred, thry, ells):
        return float(self.fn(pred, thry, ells))


# -- per-ell metrics --------------------------------------------------------- #
#
# DIRECTION IS A PROPERTY OF THE METRIC. R^2 is skill (higher is better); mse,
# tension and abs_rel_err are error (lower is better). A criterion inherits its
# metric's direction whatever the reduction, because a lower worst-case error is
# still better.

def _per_ell_r2(pred, thry, ells):
    return np.asarray(sm.decomposition(pred, thry, ells)["R2"], dtype=float)


def _per_ell_mse(pred, thry, ells):
    return np.asarray(sm.squared_error(pred, thry).mean(0), dtype=float)


def _per_ell_tension(pred, thry, ells):
    return np.asarray(sm.tension(pred, thry), dtype=float)


def _per_ell_abs_rel_err(pred, thry, ells):
    return np.nanmean(np.abs(sm.rel_err(pred, thry)), axis=0)


METRICS = {
    "r2": (_per_ell_r2, "max"),
    "mse": (_per_ell_mse, "min"),
    "tension": (_per_ell_tension, "min"),
    "abs_rel_err": (_per_ell_abs_rel_err, "min"),
}

CLIPPABLE = {"r2"}          # bounded above by 1; the others are already >= 0


# -- reductions -------------------------------------------------------------- #
#
# MIN AND MAX ARE NOT COMPARABLE ACROSS FAMILIES. ``min_r2`` is the worst
# multipole; ``min_mse`` is the best one. Likewise ``q10`` is the worst decile of
# R^2 but the best decile of mse. The literal reductions are here because they
# are unambiguous about what they compute -- but if what you mean is "worst
# case", use ``worst`` / ``worst_decile``, which resolve to the correct tail for
# the metric and stay comparable across families.

REDUCTIONS = {
    # literal: same operation regardless of direction
    "mean":         lambda v, d: np.nanmean(v),
    "median":       lambda v, d: np.nanmedian(v),
    "max":          lambda v, d: np.nanmax(v),
    "min":          lambda v, d: np.nanmin(v),
    "q10":          lambda v, d: np.nanquantile(v, 0.10),
    "q90":          lambda v, d: np.nanquantile(v, 0.90),

    # direction-aware: the bad tail, whichever end that is
    "worst":        lambda v, d: np.nanmin(v) if d == "max" else np.nanmax(v),
    "worst_decile": lambda v, d: np.nanquantile(v, 0.10 if d == "max" else 0.90),
}


def clipped_fraction(pred, thry, ells):
    """Fraction of multipoles with R^2 < 0, i.e. how much a clipped R^2
    criterion is throwing away.

    Near 0 the clipped criterion is healthy. Near 1 it pins at zero and its
    argmax is numerical dust -- check this before quoting a clipped score.
    """
    r2 = _per_ell_r2(pred, thry, ells)
    return float(np.nanmean(r2 < 0))


# -- the registry: metrics x reductions -------------------------------------- #

def _build(metric, reduction, clip_lower=False):
    """Compose one criterion. Its name is derived, never typed, so the label
    cannot drift from the behaviour."""
    per_ell, direction = METRICS[metric]
    reduce = REDUCTIONS[reduction]
    suffix = "_clipped" if clip_lower else ""

    def fn(pred, thry, ells):
        v = per_ell(pred, thry, ells)
        if clip_lower:
            v = np.clip(v, 0.0, None)
        return reduce(v, direction)

    return Criterion(f"{reduction}_{metric}{suffix}", fn, direction)


def _register():
    out = {}
    for metric in METRICS:
        for reduction in REDUCTIONS:
            key = f"{reduction}_{metric}"
            out[key] = (lambda m=metric, r=reduction: _build(m, r))
            if metric in CLIPPABLE:
                out[f"{key}_clipped"] = (
                    lambda m=metric, r=reduction: _build(m, r, clip_lower=True)
                )
    out["n_locked"] = lambda threshold=0.5: Criterion(
        f"n_locked@{threshold}",
        lambda p, t, e: float(np.sum(_per_ell_r2(p, t, e) > threshold)),
        "max",
    )
    return out


CRITERIA = _register()


# The default for compare_criteria: every reduction of R^2 (the family in use),
# plus mean and worst-case of each error metric. Built by comprehension over the
# registries rather than written out, so a new metric or reduction lands here
# without a second edit -- a hand-written default is what silently hid min_r2
# and quantile_r2 from the plots once already.
DEFAULT_CRITERIA = (
    [f"{r}_r2" for r in REDUCTIONS]
    + [f"{r}_r2_clipped" for r in ("mean", "median", "worst_decile")]
    + [f"{r}_{m}" for m in METRICS if m != "r2" for r in ("mean", "median", "worst")]
)


def make_criterion(spec, **kwargs):
    """Build a ``Criterion`` from a registry name, or pass one through.

    A bare callable is also accepted and assumed to be maximized; wrap it in a
    ``Criterion`` yourself to say otherwise.
    """
    if isinstance(spec, Criterion):
        return spec
    if callable(spec) and not isinstance(spec, str):
        return Criterion(getattr(spec, "__name__", "custom"), spec, "max")
    try:
        return CRITERIA[spec](**kwargs)
    except KeyError:
        raise ValueError(
            f"unknown criterion {spec!r}; choose from {sorted(CRITERIA)} "
            "or pass a Criterion"
        ) from None


def criteria_for(metric=None, reduction=None, clipped=None, within=None):
    """Filter the registry by metric, reduction, or clipped-ness.

    ``criteria_for(metric="mse")`` gives every reduction of mse;
    ``criteria_for(reduction="worst")`` gives the worst case of every metric.
    Returns names, ready to hand to ``compare_criteria`` or
    ``plot_criterion_curves``.

    ``within`` restricts the filter to an existing comparison's keys (e.g. a
    dict from ``compare_criteria``) instead of the full registry. Use this when
    filtering the result of a comparison that was scored with fewer criteria
    than ``CRITERIA`` -- such as the ``DEFAULT_CRITERIA`` subset -- so the
    filter can never name something that was never computed.
    """
    keys = CRITERIA if within is None else within
    out = []
    for key in keys:
        if key == "n_locked" or key.startswith("n_locked@"):
            continue
        is_clipped = key.endswith("_clipped")
        base = key[: -len("_clipped")] if is_clipped else key
        red, _, met = base.partition("_")
        if metric is not None and met != metric:
            continue
        if reduction is not None and red != reduction:
            continue
        if clipped is not None and is_clipped != clipped:
            continue
        out.append(key)
    return out


# --------------------------------------------------------------------------- #
# stage 2: scoring and selection
# --------------------------------------------------------------------------- #

def score_sweep(sweep, criterion="mean_r2", ell_max=None, **criterion_kwargs):
    """Stage 2. Reduce a sweep to a ``{degree: score}`` curve.

    Cheap relative to stage 1, so call it repeatedly with different criteria.
    ``ell_max`` masks the scoring range only; the sweep's smoothed spectra are
    untouched.
    """
    crit = make_criterion(criterion, **criterion_kwargs)
    ells = np.asarray(sweep.ells)
    keep = slice(None) if ell_max is None else (ells <= ell_max)

    ells_used = ells[keep]
    if ells_used.size == 0:
        raise ValueError(f"ell_max={ell_max} masked out every multipole")
    thry = sweep.thry[:, keep]

    return {d: crit(sweep.at(d)[:, keep], thry, ells_used) for d in sweep.degrees}


def select_degree(curve, criterion="mean_r2", rule="best", tol=1e-3, **criterion_kwargs):
    """Pick a degree from a ``{degree: score}`` curve.

    ``rule='best'``    argmax or argmin, per the criterion's direction. The
                       method in use.
    ``rule='plateau'`` smallest degree whose score is within ``tol`` of the
                       best. The parsimonious alternative for a flat curve.

    ``criterion`` is needed only for its direction; pass the same one used to
    build the curve.
    """
    if not curve:
        return None
    crit = make_criterion(criterion, **criterion_kwargs)
    better_is_more = crit.direction == "max"

    best = max(curve.values()) if better_is_more else min(curve.values())
    if rule == "best":
        return max(curve, key=curve.get) if better_is_more else min(curve, key=curve.get)
    if rule == "plateau":
        within = (
            (lambda s: s >= best - tol) if better_is_more
            else (lambda s: s <= best + tol)
        )
        return min(d for d, s in curve.items() if within(s))
    raise ValueError(f"unknown rule {rule!r}; use 'best' or 'plateau'")


def compare_criteria(sweep, criteria=None, ell_max=None, rule="best", tol=1e-3):
    """Score one cached sweep under several criteria at once.

    Returns ``{criterion_name: {"curve": ..., "direction": ..., "best": ...,
    "plateau": ..., "model": ..., "split": ...}}``. The handle for deciding what
    the criterion should be: one expensive sweep, many cheap answers, side by
    side.

    Defaults to ``DEFAULT_CRITERIA``, which is itself built by comprehension
    over the metric and reduction registries -- so a newly registered metric or
    reduction reaches these plots without a second edit. The full registry is
    large (every metric x every reduction); use ``criteria_for`` to slice it,
    e.g. ``compare_criteria(sw, criteria_for(metric="mse"))``.

    Pass an explicit ``criteria`` list to narrow it, or to include a
    parameterized variant: ``[make_criterion("quantile_r2", q=0.05)]``.
    """
    criteria = list(criteria if criteria is not None else DEFAULT_CRITERIA)
    out = {}
    for spec in criteria:
        crit = make_criterion(spec)
        curve = score_sweep(sweep, crit, ell_max=ell_max)
        out[crit.name] = {
            "curve": curve,
            "direction": crit.direction,
            "best": select_degree(curve, crit, rule="best"),
            "plateau": select_degree(curve, crit, rule="plateau", tol=tol),
            "model": sweep.model,
            "split": sweep.split,
        }
    return out


def format_comparison(comparison):
    """A printable table from ``compare_criteria``, for a notebook cell.

    ``spread`` is max minus min across the whole curve: when it is small, the
    chosen degree is being picked out of noise, whatever the criterion says.
    """
    lines = [
        f"{'criterion':<20} {'dir':<4} {'best':>5} {'plateau':>8} "
        f"{'score@best':>12} {'spread':>10}"
    ]
    for name, r in comparison.items():
        vals = list(r["curve"].values())
        lines.append(
            f"{name:<20} {r['direction']:<4} {r['best']:>5} {r['plateau']:>8} "
            f"{r['curve'][r['best']]:>12.4g} {max(vals) - min(vals):>10.3g}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# the result artifact
# --------------------------------------------------------------------------- #

@dataclass
class DegreeSelection:
    """Chosen degrees plus the curves that justify them.

    ``degrees`` maps model -> chosen degree, or None for a model that is not
    smoothed at all. None rather than a sentinel integer: an unsmoothed model
    has no degree, and giving it a placeholder that must exist as a valid dict
    key is what produced the ``if model != "obs2ps" else 10`` idiom in the
    original analysis.

    Passes straight to ``data_helper.smooth_library`` as ``sel.degrees``.
    """

    degrees: dict
    score_curves: dict
    use_ells: str = ""
    score_ell_max: dict = field(default_factory=dict)
    criterion: str = "mean_r2"
    rule: str = "best"
    smoother: str = "polyfit"
    selection_split: str = ""
    ill_conditioned: dict = field(default_factory=dict)
    epoch: str = ""

    def degree_for(self, model):
        if model not in self.degrees:
            raise KeyError(
                f"no degree recorded for {model!r}; known models: "
                f"{sorted(self.degrees)}"
            )
        return self.degrees[model]

    def plateau_degree(self, model, tol=1e-3):
        curve = self.score_curves.get(model) or {}
        if not curve:
            return None
        return select_degree(curve, self.criterion, rule="plateau", tol=tol)

    def to_dict(self):
        """A plain JSON-serializable dict. No file I/O -- the Asset's Handler
        owns that, and the encoding choices with it.

        Degree keys are stringified because JSON object keys must be strings;
        ``from_dict`` converts them back.
        """
        payload = asdict(self)
        payload["score_curves"] = {
            m: {str(d): s for d, s in c.items()} for m, c in self.score_curves.items()
        }
        return payload

    @classmethod
    def from_dict(cls, payload):
        """Rebuild from ``to_dict`` output, or from anything that round-tripped
        it. Degree keys are coerced back to int, so a format that preserved them
        as ints (yaml, pickle) also works."""
        payload = dict(payload)
        payload["score_curves"] = {
            m: {int(d): s for d, s in c.items()} for m, c in payload["score_curves"].items()
        }
        return cls(**payload)


def select_from_sweeps(
    sweeps, criterion="mean_r2", rule="best", tol=1e-3,
    ell_max=None, unsmoothed=(), use_ells="", epoch="",
    **criterion_kwargs
):
    """Build a ``DegreeSelection`` from ``{model: SmoothingSweep}``.

    ``ell_max`` is a scalar for every model, or ``{model: ell_max or None}``.
    ``unsmoothed`` models get degree None and no curve, so the result covers
    every model the figures will ask about.
    """
    crit = make_criterion(criterion, **criterion_kwargs)
    chosen, curves, masks, ill = {}, {}, {}, {}
    split = ""

    for model in list(unsmoothed):
        chosen[model], curves[model], masks[model] = None, {}, None

    for model, sweep in sweeps.items():
        if model in unsmoothed:
            continue
        m_ell_max = ell_max.get(model) if isinstance(ell_max, dict) else ell_max
        curve = score_sweep(sweep, crit, ell_max=m_ell_max)
        curves[model] = curve
        chosen[model] = select_degree(curve, crit, rule=rule, tol=tol)
        masks[model] = m_ell_max
        ill[model] = sweep.ill_conditioned
        split = split or sweep.split

    return DegreeSelection(
        degrees=chosen, score_curves=curves, criterion=crit.name, rule=rule,
        smoother=next(iter(sweeps.values())).smoother if sweeps else "polyfit",
        selection_split=split, score_ell_max=masks, ill_conditioned=ill,
        use_ells=use_ells, epoch=epoch,
    )


# --------------------------------------------------------------------------- #
# diagnostic
# --------------------------------------------------------------------------- #

def _resolve_criteria(comparison, criteria):
    """Names to plot, from a subset spec, in the order given.

    Accepts display names ('mean_r2'), ``Criterion`` objects, or bare registry
    keys for parameterized criteria -- 'n_locked' matches the stored
    'n_locked@0.5'. Raises on a name that is not in this comparison, rather than
    silently plotting fewer panels than asked for.
    """
    if criteria is None:
        return list(comparison)

    chosen, missing = [], []
    for spec in criteria:
        name = getattr(spec, "name", spec)
        if name in comparison:
            chosen.append(name)
            continue
        hits = [k for k in comparison if k.split("@", 1)[0] == name]
        if hits:
            chosen.extend(hits)
        else:
            missing.append(name)

    if missing:
        raise KeyError(
            f"not in this comparison: {missing}. This usually means the "
            f"comparison was scored with a narrower criteria list (e.g. "
            f"compare_criteria's default) than what you're filtering for. "
            f"Pass criteria_for(..., within=comparison) to filter only what "
            f"was actually computed, or rebuild with "
            f"compare_criteria(sweep, CRITERIA) to score everything. "
            f"Available here: {sorted(comparison)}"
        )
    return list(dict.fromkeys(chosen))          # dedupe, keep order


def plot_criterion_curves(comparison, criteria=None, label=None, tol_frac=0.02,
                          ncols=3, figsize=None):
    """Small multiples: one panel per criterion, from ``compare_criteria``.

    ``criteria`` selects and orders a subset of what the comparison already
    holds -- scoring is cheap but not free, so compute broadly once and choose
    what to look at here:

        comp = compare_criteria(sweep, CRITERIA)          # all of them, once
        plot_criterion_curves(comp, criteria_for(metric="mse"))
        plot_criterion_curves(comp, ["mean_r2", "worst_decile_r2"])

    ``label`` is the figure title, defaulting to the model (and split) recorded
    by ``compare_criteria``.

    Criteria live on wildly different scales (mean_r2 in units of order 1,
    mean_mse in the millions), so each panel gets its own axis. That makes
    autoscaling a trap: a curve flat to three decimal places is drawn just as
    dramatically as a real one. Two things counter it --

      * each panel's title carries the spread (max minus min over the whole
        curve), so a tiny number tells you the shape is noise;
      * a shaded band marks every degree scoring within ``tol_frac`` of the
        curve's own range from the best. A wide band means many degrees are
        effectively tied and the argmax is arbitrary.

    ``tol_frac`` is a fraction of the curve's range, not an absolute tolerance,
    precisely because the scales differ. Note it inherits the same weakness as
    ``spread``: one catastrophic degree at the low end sets the scale for
    everything, so sweep from a sensible floor.
    """
    import matplotlib.pyplot as plt

    names = _resolve_criteria(comparison, criteria)
    if label is None and names:
        first = comparison[names[0]]
        model, split = first.get("model", ""), first.get("split", "")
        label = ", ".join(p for p in (model, split) if p)

    nrows = -(-len(names) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize or (4.2 * ncols, 3.1 * nrows), squeeze=False
    )

    for ax, name in zip(axes.ravel(), names):
        r = comparison[name]
        curve = r["curve"]
        ds = sorted(curve)
        vals = np.array([curve[d] for d in ds], dtype=float)
        spread = float(np.nanmax(vals) - np.nanmin(vals))
        best_d = r["best"]
        best_v = curve[best_d]

        ax.plot(ds, vals, marker=".", ms=5, lw=1.2, color="C0")

        # every degree effectively tied with the best
        band = tol_frac * spread
        tied = (
            vals >= best_v - band if r["direction"] == "max"
            else vals <= best_v + band
        )
        if tied.sum() > 1:
            ax.fill_between(ds, *ax.get_ylim(), where=tied, alpha=0.13,
                            color="C1", step="mid", label=f"tied ({tied.sum()})")
        ax.axvline(best_d, color="C3", ls="--", lw=1, label=f"best M={best_d}")
        if r["plateau"] != best_d:
            ax.axvline(r["plateau"], color="C2", ls=":", lw=1,
                       label=f"plateau M={r['plateau']}")

        ax.set_title(f"{name}  ({r['direction']})\nspread {spread:.3g}", fontsize=9)
        ax.set_xlabel("$M$", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, loc="best", framealpha=0.8)

    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    if label:
        fig.suptitle(label, fontsize=11)
    fig.tight_layout()
    return fig


def plot_degree_choices(comparisons, criteria=None, ax=None, figsize=(8, 4)):
    """Which degree each criterion picks, for one or more models.

    ``comparisons`` maps model -> the dict returned by ``compare_criteria``.
    ``criteria`` selects and orders a subset, as in ``plot_criterion_curves``.
    Criteria run down the y-axis, chosen degree along x, one colour per model.
    A vertical stack means the criteria agree; a scatter across the range means
    the choice is being made by the criterion rather than by the data.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if criteria is None:
        names = sorted({n for c in comparisons.values() for n in c})
    else:
        names = list(dict.fromkeys(
            n for c in comparisons.values() for n in _resolve_criteria(c, criteria)
        ))
    ypos = {n: i for i, n in enumerate(names)}

    for i, (model, comp) in enumerate(sorted(comparisons.items())):
        xs = [comp[n]["best"] for n in names if n in comp]
        ys = [ypos[n] for n in names if n in comp]
        ax.scatter(xs, ys, s=55, label=model, color=f"C{i}", zorder=3,
                   alpha=0.85, edgecolors="white", linewidths=0.6)
        xs_p = [comp[n]["plateau"] for n in names if n in comp]
        ax.scatter(xs_p, ys, s=22, color=f"C{i}", marker="|", zorder=2)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("chosen degree $M$")
    ax.set_title("Degree chosen by criterion (dot = best, tick = plateau)", fontsize=10)
    ax.grid(axis="x", alpha=0.25, zorder=0)
    ax.legend(fontsize=8, loc="best")
    ax.invert_yaxis()
    return ax


def plot_score_curves(selection, ax=None, mark_plateau=True, tol=1e-3):
    """Score vs degree for every swept model. The justification for a chosen
    degree travels with the choice, so it lives here rather than in a figure
    module.

    matplotlib is imported inside the function: nothing else here needs it, and
    a sweep should run without a display stack.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    for model, curve in selection.score_curves.items():
        if not curve:
            continue
        ds = sorted(curve)
        ax.plot(ds, [curve[d] for d in ds], label=model, marker=".", ms=4)
        ax.axvline(selection.degrees[model], ls="--", lw=0.7, alpha=0.5)
        if mark_plateau:
            p = selection.plateau_degree(model, tol=tol)
            if p is not None and p != selection.degrees[model]:
                ax.axvline(p, ls=":", lw=0.7, alpha=0.5)

    ax.set_xlabel("polynomial degree $M$")
    ax.set_ylabel(selection.criterion)
    ax.set_title(
        f"Degree selection ({selection.criterion}, {selection.smoother})"
        + (f", split = {selection.selection_split}" if selection.selection_split else "")
    )
    ax.legend(loc="best")
    return ax
