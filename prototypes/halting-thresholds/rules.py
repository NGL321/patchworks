"""The four halting rules, as pure functions over what a run produces.

**Throwaway.** Built for [issue #156](https://github.com/NGL321/patchworks/issues/156).

One function per live-readable entry of the falsification register
([#147](https://github.com/NGL321/patchworks/issues/147)). Each returns a
:class:`Verdict` — fired or not, at which tick, and on what reading — so
``report.py`` can ask the only question that matters of a halting rule: *would
this have fired at the right tick, for the right reason?*

## The constant that is not there

`08-the-acceptance-demo.md` refuses to pre-register a ratio threshold, on the
ground that it "would be a number invented before anything was trained", and
pre-registers **orderings** instead. That refusal is binding here. Every rule
below is written so that its load-bearing constant is either

* a **count** (a window, a dwell, a sample floor) — a statement about patience,
  not about the world; or
* a **ratio of the signal to itself** (before a hold against after it, one
  regime's spread against another's) — scale-free by construction; or
* **already fixed** by an earlier ticket, and merely cited here.

Nothing below invents a level. Where a level would be needed, the rule is
written to not need one, and that is the substance of this prototype.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Verdict:
    fired: bool
    tick: int | None
    reading: str


# --------------------------------------------------------------------------
# entry 1 — a linear chart cannot hold an either/or (#151)
# --------------------------------------------------------------------------


def naive_slope(
    error: np.ndarray, *, window: int, min_decline: float, floor: float
) -> Verdict:
    """The tempting rule, and the one to argue against.

    Over a trailing ``window``, halt if the error fell by less than
    ``min_decline`` of its own level while sitting above ``floor``.

    Both of its constants are the wrong kind. ``floor`` is an invented level —
    exactly what `08-the-acceptance-demo.md` forbids. ``min_decline`` at a
    fixed ``window`` cannot tell an irreducible residual from one decaying with
    a time constant longer than the window, and no choice of window fixes that,
    because the run's own learning time constant is unknown until it is over.
    Kept here to be shown failing, not to be used.
    """
    # **Non-overlapping windows, on a fixed schedule.** Re-testing at every tick
    # is a multiple-comparison problem and not a small one: a 3000-tick run
    # offers ~2600 overlapping chances to cross any bar, so a 3-sigma test on a
    # healthy cell fires as a matter of routine. The prototype hit this and it
    # is the reason the halt is evaluated on a schedule rather than continuously
    # — one decision per window, ~7 per run, which a 3-sigma bar can carry.
    for end in range(window, len(error) + 1, window):
        w = error[end - window:end]
        # The decline is read on the content *above* the reference level, not on
        # the raw error. Any converging curve's fractional decline tends to zero
        # as it settles onto its floor, so a raw reading eventually nominates
        # every healthy cell — the prototype ran into exactly that.
        first = w[: window // 2].mean() - floor
        last = w[window // 2:].mean() - floor
        if first <= 0:
            continue
        # The excess over the reference level must be distinguishable from it
        # before its slope means anything. Once a curve settles onto the
        # baseline the excess is noise, and a fractional decline computed on
        # noise wanders freely — the prototype watched a healthy cell get
        # halted that way at tick 2558. The bar uses the window's own observed
        # spread, so it is still a ratio of the signal to itself.
        stderr = w.std(ddof=1) / np.sqrt(window / 2.0)
        if last <= 3.0 * stderr:
            continue
        decline = (first - last) / first
        if decline < min_decline:
            return Verdict(
                True, end - 1, f"decline {decline:+.3f} over {window} < "
                f"{min_decline}, level {last:.3f} > floor {floor}"
            )
    return Verdict(False, None, "never flat-and-high inside a window")


def honoured_decline(*, window: int, tau_max: float) -> float:
    """The nomination slope, derived from the budget rather than chosen.

    This is the answer to the ticket's hardest question — *how long must a run
    go before "does not fall with learning" is a claim rather than an
    impatience* — and it turns out not to be a length at all.

    :func:`naive_slope` compares two half-window means, whose centres are
    ``W/2`` apart, so the decline it measures for an exponential with time
    constant ``tau`` is ``1 - exp(-(W/2)/tau)`` — **not** ``1 - exp(-W/tau)``.
    Deriving the constant from the window instead of from the estimator was the
    prototype's own first mistake, and it halted a cell learning comfortably
    inside the budget. The lever and the reading have to be the same quantity.

    With that fixed, refusing to nominate anything declining faster than
    ``min_decline`` is exactly the statement *this run will wait for learning
    with a time constant up to* ``tau_max``, and no longer. The pair
    ``(window, min_decline)`` carries no information the pair
    ``(window, tau_max)`` does not.

    So the constant is set by the **compute budget** — which #147 said all along
    the thresholds depend on, and which is knowable without a run — rather than
    by the data. Pre-register ``tau_max`` as a fraction of the run's tick
    budget, and the slope follows.
    """
    return float(1.0 - np.exp(-(window / 2.0) / tau_max))


def detectable(*, window: int, min_decline: float, noise: float) -> bool:
    """Is a decline of ``min_decline`` visible over ``window`` at this noise?

    The two half-window means each carry a standard error of
    ``noise / sqrt(window / 2)``, so their difference carries ``sqrt(2)`` times
    that. Requiring a 3-sigma separation is the whole check.

    ``noise`` is the one genuinely scale-bound input in this file, and it enters
    as a **feasibility check on a chosen constant**, never as the constant
    itself. If a real run's noise makes the pre-registered ``tau_max``
    undetectable, the honest response is to shorten ``tau_max`` or lengthen the
    window before the run — not to move the threshold during it.
    """
    stderr = noise * np.sqrt(2.0 / (window / 2.0))
    return bool(min_decline > 3.0 * stderr)


def hold_confirmed(
    error: np.ndarray,
    regime,
    rng: np.random.Generator,
    *,
    window: int,
    min_decline: float,
    hold: int,
    survives: float,
    baseline: float,
) -> Verdict:
    """Entry 1's rule: the slope only nominates; the **quiescent hold** decides.

    ADR-0004's disambiguator, used as the halt rather than as a post-hoc
    reading. Four gates, in cost order:

    1. **Construction.** If #49's box-counting criterion failed on this edge, or
       #141's shared ``colspan(D)`` direction is present, the residual has a
       booked cause that is not entry 1. Read forwards, costs nothing, and rules
       the edge out of this halt entirely.
    2. **Baseline.** ADR-0004 names two comparisons and the quiescent hold is
       only the second: the residual must first clear
       `06-graph-topology.md`'s **topology-only baseline**. That baseline is a
       *measured* construction-time quantity, which is what lets this rule have
       a reference level without inventing one — and it is not optional. A cell
       that has converged onto a small non-draining residual is a cell that is
       working, and without this gate the hold happily halts on it.
    3. **Nomination.** A trailing ``window`` in which the error did not fall by
       ``min_decline``, the latter derived from the budget by
       :func:`honoured_decline`. Deliberately loose — a false nomination costs
       one hold, not a run.
    4. **Confirmation.** Hold the world still for ``hold`` ticks. Lag drains;
       curvature does not. Halt only if at least ``survives`` of the nominated
       level is still standing when the hold ends.

    ``survives`` is a ratio of the signal to itself — the level after the hold
    against the level before it — so it carries no units and needs no
    calibration run.

    **What the hold does not do.** It cannot separate *irreducible* from
    *reducible but slower than the budget*, because nothing learns during a
    hold either way. That separation is gate 3's job and gate 3 does it by
    definition rather than by measurement: past ``tau_max`` the two are the same
    claim, and the register's entry is a statement about this budget.
    """
    if regime.selfint_at_construction:
        return Verdict(
            False, None, "#49 self-intersection flagged at construction; "
            "residual has a booked cause that is not entry 1"
        )
    if regime.gauge_at_construction:
        return Verdict(
            False, None, "#141 shared-direction residual present; entry 6, "
            "read forwards, not a halt"
        )

    nomination = naive_slope(
        error, window=window, min_decline=min_decline, floor=baseline
    )
    if not nomination.fired:
        return Verdict(False, None, "never nominated above the baseline")

    at = nomination.tick
    assert at is not None
    level = float(error[max(0, at - window // 2): at + 1].mean())
    held = regime.under_hold(level, hold, rng)
    remaining = float(held[-max(1, hold // 10):].mean())
    ratio = remaining / level if level > 0 else 0.0

    if ratio >= survives:
        return Verdict(
            True, at + hold,
            f"nominated at {at}, {ratio:.2f} of the level survived a "
            f"{hold}-tick hold (>= {survives})"
        )
    return Verdict(
        False, None,
        f"nominated at {at}, but only {ratio:.2f} survived the hold "
        f"(< {survives}) — lag, and ADR-0007 tolerates it"
    )


# --------------------------------------------------------------------------
# entry 2 — open-loop rollout decays into the dominant subspace (#144)
# --------------------------------------------------------------------------


def rollout_horizon(singular_values: np.ndarray, *, tol: float) -> float:
    """Ticks until non-dominant content has decayed to ``tol`` of the dominant.

    Not a halt, and this is the finding rather than a caveat. #147 already
    records that this horizon is "computable per cell from the singular-value
    gap, so knowable rather than discovered" — which makes it a **construction
    check**, read once from ``K``, not a curve watched during a run.

    With ``s1 >= s2`` the ratio of the second mode to the first after ``h``
    steps is ``(s2/s1)**h``, so the horizon is ``log(tol) / log(s2/s1)``.
    """
    s = np.sort(np.asarray(singular_values, dtype=float))[::-1]
    if len(s) < 2 or s[0] <= 0:
        return float("inf")
    gap = s[1] / s[0]
    if gap <= 0:
        return 0.0
    if gap >= 1.0:
        return float("inf")
    return float(np.log(tol) / np.log(gap))


def horizon_sufficient(
    singular_values: np.ndarray, *, required: int, tol: float
) -> Verdict:
    """Compare a cell's rollout horizon to the loop it has to serve.

    ``required`` comes from `06-graph-topology.md` by way of
    `08-the-acceptance-demo.md`: the somatomotor reflex loop is **three ticks**,
    a correction needing visual context is four hops out and back. So the number
    is supplied by the graph, not invented here.
    """
    h = rollout_horizon(singular_values, tol=tol)
    if h < required:
        return Verdict(
            True, 0, f"horizon {h:.1f} ticks < {required} required by the loop"
        )
    return Verdict(False, None, f"horizon {h:.1f} ticks >= {required}")


# --------------------------------------------------------------------------
# entry 4 — a cell that amplifies instead of forgetting (#140)
# --------------------------------------------------------------------------


def projection_binding(
    pre_projection_sigma_max: np.ndarray,
    *,
    dwell: int,
    fraction: float,
    burn_in: int,
) -> Verdict:
    """Halt when the band has to be *enforced* rather than merely respected.

    #140 puts ``sigma_max(K)`` in ``[1/rho_K, 1]`` and restores it by projection
    after every learning step, so the post-projection value is never out of
    band and reading it says nothing. The live quantity is the **pre**-projection
    one: how often learning pushes a cell above 1 and has to be pulled back.

    The band's upper face is exactly 1 and #140 fixed it, so the level is cited,
    not invented. What is left is a count — ``dwell`` consecutive ticks with at
    least ``fraction`` of the fleet binding — which is a statement about how
    long amplification pressure must persist before it is a property of the
    build rather than a transient of learning.

    ``burn_in`` is the other count, and the prototype found it the hard way.
    #138 initialises ``K = aI``, so the fleet's **first** excursion against the
    band is the selection rig's settling, not a cost: every cell binds at once
    and a dwell shorter than that transient halts on the architecture doing
    what it was designed to do. The exclusion is legitimate because the settling
    length is a construction quantity the selection rig already produces — not a
    window chosen once the data was in view.

    ``pre_projection_sigma_max`` is ``(ticks, cells)``.
    """
    series = np.asarray(pre_projection_sigma_max)[burn_in:]
    binding = (series > 1.0).mean(axis=1)
    run = 0
    for t, frac in enumerate(binding):
        run = run + 1 if frac >= fraction else 0
        if run >= dwell:
            return Verdict(
                True, t + burn_in,
                f"{frac:.0%} of cells bound the band for {dwell} ticks "
                f"after a {burn_in}-tick burn-in"
            )
    return Verdict(
        False, None,
        f"band never bound by {fraction:.0%} of the fleet for {dwell} "
        f"consecutive ticks past burn-in"
    )


# --------------------------------------------------------------------------
# entry 5 — one global band spanning two regimes (#146, on #149's ground)
# --------------------------------------------------------------------------


def _iqr(x: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(x, 25)), float(np.percentile(x, 75))


def regime_split(
    contact_error: np.ndarray,
    free_error: np.ndarray,
    *,
    windows: int,
    min_samples: int,
) -> Verdict:
    """Halt when contact and free motion separate **and keep separating**.

    Deliberately the acceptance demo's own instrument. `08` establishes
    non-overlapping interquartile ranges as this project's pre-registered way to
    claim two populations differ without inventing a ratio, and the same move
    works here: one global band straining across two natural rates shows up as
    per-cell error whose contact and free distributions pull apart.

    Two gates, because separation alone is not the cost. A cell may simply find
    contact harder. The cost is a band that cannot hold both, which gets
    **worse**, so the rule also requires the gap to widen monotonically across
    ``windows`` consecutive windows. That is an ordering, not a level.

    ``min_samples`` guards the quartiles, since contact is intermittent.
    """
    contact_error = np.asarray(contact_error, dtype=float)
    free_error = np.asarray(free_error, dtype=float)
    n = min(len(contact_error), len(free_error))
    if n < min_samples * windows:
        return Verdict(False, None, "not enough contact samples to read")

    size = n // windows
    gaps: list[float] = []
    for w in range(windows):
        c = contact_error[w * size:(w + 1) * size]
        f = free_error[w * size:(w + 1) * size]
        if len(c) < min_samples or len(f) < min_samples:
            return Verdict(False, None, "a window fell under min_samples")
        c_lo, _ = _iqr(c)
        _, f_hi = _iqr(f)
        gaps.append(c_lo - f_hi)

    separated = all(g > 0 for g in gaps)
    widening = all(b > a for a, b in zip(gaps, gaps[1:]))
    if separated and widening:
        return Verdict(
            True, n - 1,
            f"IQRs disjoint in every window and the gap widened "
            f"{gaps[0]:.3f} -> {gaps[-1]:.3f}"
        )
    return Verdict(
        False, None,
        f"separated={separated} widening={widening}; gaps "
        + ", ".join(f"{g:+.3f}" for g in gaps)
    )
