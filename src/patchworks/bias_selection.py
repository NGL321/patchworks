"""Bias selection: the spread, the fold margin, and the go/no-go.

The construction rig that places each cell's timescale, promoted from
`prototypes/regional-spectra` (`spread_pilot.py`, `selection_sweep.py`). It runs
**before anything is trained** and it is the falsification condition for
`docs/adr/0005-timescale-is-persistence-not-a-schedule.md`, so it can kill the
timescale mechanism cheaply.

Two quantities stay permanently distinguished, and the whole module is arranged
around the distinction (`docs/spec/05-timescales.md`, *The mechanism*):

* the **regional spectrum** is per-tick — the local Jacobian of whichever
  activation region the cell occupies this tick, re-drawn as its chart crosses
  folds — and is never a cell attribute;
* a cell's **effective timescale** is the central tendency of the distribution
  those draws come from, and *that distribution is what the biases select*.

The construction has four parts, and they are as much a constraint on the body
as `n = 32` and `k = 12`:

1. **`sigma_w^2` is set for containment and never asked to buy spread.** It is
   :data:`patchworks.body.DEFAULT_WEIGHT_VARIANCE` and nothing here touches it.
2. **The spread is imposed by selection, not by drawing.** :func:`sweep` draws
   candidate bias vectors and measures the timescale each produces;
   :func:`select` keeps a set covering the target band and discards the rest.
   Nothing is added to the architecture — no rate is stored, and a cell's rate
   is still whatever its region gives it.
3. **The target is a range in ticks derived from the acceptance demo's
   perturbation horizons** (:class:`DemoHorizons`). The derivation is fixed by
   the spec; the numbers are not, and are an input to this module rather than a
   constant inside it.
4. **The slow end is capped by measured contraction `lambda`**, not by a `rho`
   ceiling: :meth:`Sweep.slow_cap` is the slowest `tau` for which realised
   `lambda` stays negative by a stated safety factor, a number this run produces
   per body.

Selected timescales are **assigned by level in overlapping bands**
(:meth:`TargetRange.bands`). Banding is a construction choice and it costs a
piece of evidence: the depth-timescale correspondence is built rather than
found, so it can no longer be cited as the mechanism working. What stays
falsifiable is behavioural.

The module also carries the **fold margin** check (:func:`fold_margin_check`) —
same sweep, same afternoon. Since #138 the margin is read from **`encode`
alone**, because `encode` is the body's only nonlinearity and so the only map
that has folds at all. The check is **demoted rather than deleted** (#140): it
is no longer a bound on `gamma`, whose constraint is exactly two things —
capped at 1.0 globally, and ADR-0010's provable
`lambda_max(sum_e F^T F) <= rho^2 deg(v)`.

**Demoted again by #160, and this time out of construction** (ADR-0019).
Neither side of the bound holds still: the standing offset falls 144x through a
run (#158), and the folds themselves slide, because their positions are the
per-cell biases the prediction rule trains. So what this module produces is a
**nomination** — the cap a body's draw permits, before anything runs — and
:class:`patchworks.tick.FoldRead` is what decides, live, on the run that
actually happens. ADR-0005's falsification duty travels with the verdict: it is
measured dwell that can kill the timescale mechanism, and the live
margin-against-offset comparison that attributes a killed cell to
reconciliation rather than to its own dynamics.

The rig also produces **`a`**, the scalar in `K = a.I` at construction
(:func:`operator_scale_rule`) — a fifth part of the construction, and a number
this run produces per body exactly as :meth:`Sweep.slow_cap` is.

**The go/no-go is read as a shape check.** The body's widths are fixed but
nothing is trained, so the sweep runs plausible chart and stalk sequences rather
than real ones (:func:`driven_trajectory`). It establishes that the mechanism is
available. It does not produce the body's number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .body import DEFAULT_OPERATOR_SCALE, DEFAULT_RHO_K, CellBiases, CellBody
from .graph import Dome
from .restriction import GAUGE_RHO

__all__ = [
    "DEFAULT_BURN_IN",
    "DEFAULT_DRAWS",
    "DEFAULT_DRIVE_CORRELATION",
    "DEFAULT_OVERLAP",
    "DEFAULT_SAFETY_FACTOR",
    "DEFAULT_TICKS",
    "MAP_NORM_BOUND",
    "TAU_QUANTILES",
    "Band",
    "DemoHorizons",
    "FoldMarginCheck",
    "GoNoGo",
    "Measurement",
    "Selection",
    "Sweep",
    "TargetRange",
    "driven_trajectory",
    "fold_margin_check",
    "go_no_go",
    "measure",
    "select",
    "sweep",
]

#: @type chosen
#: @flexibility unknown, and this register's flagship unknown: #178 found a rig tick count silently deciding a result
#: @warrant here
#: Ticks of driven trajectory each candidate is measured over. The rig's
#: (`selection_sweep.trajectory_lambda`), long enough for the `lambda` estimate
#: to settle and short enough that 64 of them per candidate stay an afternoon.
DEFAULT_TICKS = 64

#: @type chosen
#: @flexibility unknown
#: @warrant here
#: Ticks run before measurement starts, so that the chart is measured on the
#: trajectory the drive puts it on rather than on the way there.
DEFAULT_BURN_IN = 16

#: @type chosen
#: @flexibility the arm the dwell measurement is most sensitive to; benchmarks/timescale_selection.py sweeps it rather than trusting it
#: @warrant docs/spec/05-timescales.md
#: Correlation time, in ticks, of the node stalk sequence the cell is driven
#: with. **A stand-in, and the one the dwell arm is most sensitive to**: nothing
#: is trained, so there is no real message stream to drive a cell with. The rig
#: resampled the operating point every tick, which is the *no-dwell* extreme —
#: it flatters `lambda` and makes region dwell identically one tick, so the
#: precondition `05-timescales.md` states could not be read off it at all. A
#: correlated drive is what makes dwell a measurement rather than a constant;
#: the benchmark sweeps this rather than trusting it.
DEFAULT_DRIVE_CORRELATION = 8.0

#: @type stipulated
#: @flexibility free as a convention, but nothing the rig reads is comparable across two values of it, which is what fixing it buys
#: @warrant here
#: Standard deviation of the draws the driven trajectory's Ornstein-Uhlenbeck
#: walk is built from. The innovation scaling is `sqrt(1 - retention^2)`, so the
#: **stationary** std of the walk *is* this number -- and `driven_trajectory`'s
#: docstring already claims "unit stationary variance", which `1.0` is precisely
#: what makes true. Stipulated rather than chosen: it is the selection rig's
#: amplitude convention, not a number argued locally.
#: **Not `agent.DRIVE_ASSERTION`**, which wears the same 1.0. That is the
#: drive's asserted scalar, fixed by #137 on ADR-0009's warrant; this is the
#: amplitude of a synthetic node stalk in the selection rig, about which
#: ADR-0009 says nothing. The shared number is the trap, so the two are named
#: apart and the collision is written down here.
DEFAULT_DRIVE_SCALE = 1.0

#: @type chosen
#: @flexibility unknown; #42's rig drew 20,000 at a frozen operating point, and each draw here costs a whole trajectory
#: @warrant here
#: Candidate bias vectors drawn per sweep. #42's rig drew 20,000 at a frozen
#: operating point; each draw here costs a whole trajectory instead, so the
#: default is smaller and the benchmark says what it used.
DEFAULT_DRAWS = 4096

#: @type stipulated
#: @flexibility none as a choice of statistic: tau = -1/ln rho diverges as rho -> 1, so no moment is admissible
#: @warrant docs/spec/05-timescales.md
#: `tau` is reported as **quantiles, never moments**: `tau = -1/ln rho` diverges
#: as `rho -> 1`, so a mean is dominated by whichever draw came closest to the
#: boundary.
TAU_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)

#: @type chosen
#: @flexibility unknown
#: @warrant docs/spec/05-timescales.md, The taper's timescale gradient is a gradient in means
#: How far adjacent levels' bands overlap, as a fraction of a band's width. The
#: taper's gradient is continuous and separates levels only as distributions
#: (`05-timescales.md`, *The taper's timescale gradient is a gradient in
#: means*), so bands overlap by construction; a half-band overlap is this
#: module's choice of how much.
DEFAULT_OVERLAP = 0.5

#: @type measured
#: @flexibility not a knob: 2.6 is #27's measured one-tick non-normal amplification
#: @warrant #27
#: The safety factor the slow cap is stated with. #27 measured a **2.6x**
#: one-tick non-normal amplification, and the slow-and-stable band is thin, so
#: the factor is not decorative: a cell whose realised decay is slower than its
#: regions imply by more than this is one bias update from crossing.
DEFAULT_SAFETY_FACTOR = 2.6

#: @type derived
#: @depends_on patchworks.restriction.GAUGE_RHO
#: @flexibility none independently: it is the spec's rho, and the import is what makes disagreeing with it impossible
#: @warrant docs/adr/0010-restriction-map-scale-is-gauge-fixed.md
#: `rho` in `lambda_max <= rho^2 deg(v)`: the gauge bound on a restriction map's
#: Frobenius norm (`docs/adr/0010-restriction-map-scale-is-gauge-fixed.md`),
#: which is the same `rho` as the gauge's band edge rather than a number that
#: agrees with it -- so it is **imported** rather than restated
#: (`docs/adr/0018-a-derived-constant-is-derived-where-its-dependency-lives.md`).
#: Named apart from the spectral radius this module otherwise calls `rho`: the
#: alias is a rename, not a second declaration, and it keeps a register row
#: saying the slow cap depends on the gauge.
MAP_NORM_BOUND = GAUGE_RHO

#: `rho` this close to one is reported as this `tau` rather than as infinity.
#: An expansive region has no `tau` at all; clamping keeps it at the slow end of
#: the ordering, which is where a quantile wants it, and the expansive fraction
#: is reported separately so nothing reads the clamp as a timescale.
_RHO_CEILING = 1.0 - 1e-6



@dataclass(frozen=True)
class DemoHorizons:
    """The two numbers the target range is derived from, in ticks.

    `05-timescales.md` deliberately fixes the **derivation** and not the
    numbers: the demo is still open and is likely to grow as compositional
    behaviour is asked for. So they are an argument here rather than a constant,
    and the two readings the record supports differ by two orders of magnitude —
    see `docs/spec/08-the-acceptance-demo.md`, *What is measured*.

    * `fastest` — the horizon of the fastest perturbation the demo applies, the
      arm nudge. The recorded number is the somatomotor reflex loop's **three
      ticks** (`06-graph-topology.md`, `08-the-acceptance-demo.md`).
    * `longest` — the horizon of the longest, the retarget: how long the
      re-asserted goal has to be held for. That is a whole task, and the only
      measured task duration in the record is the achievability run's
      (`benchmarks/achievability.py`).
    """

    fastest: float
    longest: float

    def __post_init__(self) -> None:
        if self.fastest <= 0:
            raise ValueError(f"a horizon is a positive number of ticks, got {self.fastest}")
        if self.longest <= self.fastest:
            raise ValueError(
                "the longest perturbation horizon must exceed the fastest, got "
                f"{self.longest} against {self.fastest}"
            )

    @property
    def target(self) -> "TargetRange":
        """The target range: fast enough to resolve one, slow enough to outlast the other."""
        return TargetRange(fastest=self.fastest, slowest=self.longest)


@dataclass(frozen=True)
class Band:
    """One level's timescale band: the range of effective timescales it holds.

    A band is where a cell **started**, not a property it has. Nothing stores
    it, nothing re-selects, and the biases drift off it as they adapt
    (`CONTEXT.md`, *Timescale band*).
    """

    level: int
    lo: float
    hi: float

    def holds(self, tau: torch.Tensor) -> torch.Tensor:
        """`[candidates]` bool: whose effective timescale lands in this band."""
        return (tau >= self.lo) & (tau <= self.hi)

    def __str__(self) -> str:
        return f"L{self.level}: {self.lo:.1f}-{self.hi:.1f} ticks"


@dataclass(frozen=True)
class TargetRange:
    """The range of effective timescales the construction is aiming at, in ticks.

    The fast end resolves the fastest perturbation the demo applies; the slow
    end outlasts the longest. :meth:`capped_at` is where the measured `lambda`
    then has its say.
    """

    fastest: float
    slowest: float

    def __post_init__(self) -> None:
        if self.fastest <= 0 or self.slowest <= self.fastest:
            raise ValueError(
                f"a target range runs from a positive fast end upward, got "
                f"{self.fastest} to {self.slowest}"
            )

    @property
    def ratio(self) -> float:
        return self.slowest / self.fastest

    def bands(self, levels: int, *, overlap: float = DEFAULT_OVERLAP) -> tuple[Band, ...]:
        """Split the range into `levels` **overlapping** bands, log-spaced.

        Log-spaced because `tau` is a rate and the range spans orders of
        magnitude; overlapping because the taper's gradient is one in means,
        with adjacent depths overlapping on any single tick. `levels` are
        numbered from 1 outward-in, so the shallowest level gets the fastest
        band and the apex the slowest.

        `overlap` is the fraction of a band shared with each neighbour: at 0 the
        bands tile, at 0.5 each band's lower half is its neighbour's upper half.
        """
        if levels < 1:
            raise ValueError(f"the taper has at least one level, got {levels}")
        if not 0.0 <= overlap < 1.0:
            raise ValueError(f"overlap is a fraction below one, got {overlap}")
        lo, hi = math.log(self.fastest), math.log(self.slowest)
        if levels == 1:
            return (Band(level=1, lo=self.fastest, hi=self.slowest),)
        # `levels` bands of width `w`, each starting `(1 - overlap) * w` above
        # the last, together spanning the range exactly.
        width = (hi - lo) / (levels - overlap * (levels - 1))
        return tuple(
            Band(
                level=i + 1,
                lo=math.exp(lo + i * (1.0 - overlap) * width),
                hi=math.exp(lo + i * (1.0 - overlap) * width + width),
            )
            for i in range(levels)
        )


@dataclass(frozen=True)
class Measurement:
    """What one driven trajectory says about each candidate bias vector.

    Every field is `[candidates]` except :attr:`tau`, which is
    `[candidates, len(TAU_QUANTILES)]`. Nothing here is a cell attribute: it is
    what the rig measured about a *candidate*, from outside, before the graph
    exists.
    """

    tau: torch.Tensor
    """Quantiles of the per-tick regional `tau` at :data:`TAU_QUANTILES`."""

    dwell: torch.Tensor
    """Mean ticks in one activation region before the chart crosses a fold.

    The precondition (`05-timescales.md`, *The precondition: region dwell against
    `tau`*): where this is short against :attr:`effective_timescale`, the cell
    still decays at some average rate but does so by averaging over unrelated
    regions, which is a different mechanism from the one specified.
    """

    contraction: torch.Tensor
    """`lambda`, the realised contraction rate along the trajectory.

    `lim (1/T) log ||J_T ... J_1||`. **The stability object**: a cell is unstable
    when `lambda >= 0`, which is not the same as occupying a region whose
    spectral radius exceeds one.
    """

    rho_median: torch.Tensor
    """Median regional spectral radius over the regions the trajectory visited.

    The same central tendency :attr:`effective_timescale` reports, before
    `tau = -1/ln rho` is taken — and the honest form of it where `rho >= 1`,
    since an expansive region has no `tau` for a quantile to land on. Read by
    :meth:`contained` rather than the clamped `tau`, which cannot express it.
    """

    finite: torch.Tensor
    """`[candidates]` bool: whether the driven trajectory stayed finite.

    A chart that overflows makes every pre-activation NaN, which reads as *no
    unit active* — a zero Jacobian, `rho = 0`, and the fastest, most contained
    candidate the rig can report. Divergence that hard has to be surfaced by
    the falsification instrument, not absorbed into its numerical guards.
    """

    rho_max: torch.Tensor
    """Largest regional spectral radius over the regions the trajectory visited.

    `max rho < 1` is the cheap construction-time **sufficient** check, demoted
    but retained: if no region a cell occupies is expansive then no trajectory
    through them can diverge, whatever the dwell.

    Over the regions **visited**, which is a weaker statement than the record's
    "can occupy" — the regions a cell can reach are not enumerable, and a
    trajectory is what there is. Read it as the cheap check it is; `lambda` is
    what the go/no-go measures.
    """

    expansive: torch.Tensor
    """Fraction of visited regions with `rho >= 1`. Harmless where dwell is short."""

    margin: torch.Tensor
    """Median fold margin along the trajectory, over `encode`.

    Hanin & Rolnick's distance to the nearest region boundary,
    `min_i |z_i| / ||grad z_i||`. Read from `encode` alone since #138: `K` and
    `decode` are linear and have no folds, so there is no second map to
    minimise over.
    """

    ticks: int

    @property
    def effective_timescale(self) -> torch.Tensor:
        """`[candidates]`: the central tendency of each candidate's `tau` draws.

        The median, because `tau = -1/ln rho` diverges as `rho -> 1` and a mean
        would be the tail. This is the quantity the biases select and the bands
        are read against — and it is the *distribution's* centre, not a rate the
        cell has.
        """
        return self.tau[:, TAU_QUANTILES.index(0.5)]

    def contained(self, safety_factor: float = DEFAULT_SAFETY_FACTOR) -> torch.Tensor:
        """`[candidates]` bool: whose realised `lambda` stays negative with margin.

        Two conditions. `lambda < 0` — the trajectory contracts. And the
        realised timescale `-1/lambda` may not exceed the regions' own
        `tau` by more than `safety_factor`, which is what "negative *by* a stated
        safety factor" buys: a candidate whose product decays far slower than
        its regions imply is being held up by non-normal transients, and #27
        measured those at 2.6x on one tick.
        """
        if safety_factor <= 1.0:
            raise ValueError(f"the safety factor is above one, got {safety_factor}")
        contracts = self.contraction < 0.0
        realised = torch.where(
            contracts, -1.0 / self.contraction.clamp(max=-1e-12), torch.inf
        )
        tau = self.effective_timescale
        # A candidate whose median region is expansive has no `tau` at all, only
        # the clamp standing in for one, so it is refused on `rho` rather than on
        # a `tau` the clamp would let through. A candidate whose trajectory left
        # the reals is refused for the opposite reason: it reads as the fastest
        # thing here, and it is the slowest thing possible.
        return (
            contracts
            & self.finite
            & (self.rho_median < 1.0)
            & (realised <= safety_factor * tau)
        )

    def slow_cap(self, safety_factor: float = DEFAULT_SAFETY_FACTOR) -> float:
        """The slowest effective timescale that stays contained, in ticks.

        `05-timescales.md`'s fourth construction part, and **a number this run
        produces per body** rather than a constant: the slow end of the target
        range is capped here, not by a `rho` ceiling.
        """
        contained = self.contained(safety_factor)
        if not bool(contained.any()):
            return 0.0
        return float(self.effective_timescale[contained].max())


def driven_trajectory(
    body: CellBody,
    biases: CellBiases,
    *,
    ticks: int = DEFAULT_TICKS,
    burn_in: int = DEFAULT_BURN_IN,
    drive_correlation: float = DEFAULT_DRIVE_CORRELATION,
    drive_scale: float = DEFAULT_DRIVE_SCALE,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """A plausible node stalk sequence, `[burn_in + ticks, candidates, n]`.

    **Plausible, not real.** Nothing is trained, so there is no message stream to
    drive a cell with; what matters for the go/no-go is that the operating point
    **varies as it will at runtime** rather than being frozen, and that it varies
    *smoothly*, since a stalk resampled every tick makes region dwell identically
    one tick and measures the no-dwell extreme instead of the mechanism.

    An Ornstein-Uhlenbeck walk with unit stationary variance and a correlation
    time of `drive_correlation` ticks. The chart is not drawn: it is whatever the
    cell's own recurrence makes of this sequence.
    """
    if ticks < 1 or burn_in < 0:
        raise ValueError(f"got ticks={ticks}, burn_in={burn_in}")
    if drive_correlation <= 0:
        raise ValueError(f"drive_correlation is positive, got {drive_correlation}")
    shape = (biases.cells, body.shape.n)
    like = body.encode_hidden_weight
    retention = math.exp(-1.0 / drive_correlation)
    innovation = math.sqrt(1.0 - retention**2)

    def draw() -> torch.Tensor:
        return torch.empty(shape, device=like.device, dtype=like.dtype).normal_(
            0.0, drive_scale, generator=generator
        )

    stalk = draw()
    out = torch.empty((burn_in + ticks, *shape), device=like.device, dtype=like.dtype)
    for t in range(burn_in + ticks):
        out[t] = stalk
        stalk = retention * stalk + innovation * draw()
    return out


def _map_jacobian(
    hidden_weight: torch.Tensor,
    output_weight: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """The Jacobian of one map in the activation region `active` selects.

    `[candidates, d_y, d_x]`: the map is exactly affine on that region, so its
    Jacobian is `W_out diag(active) W_hidden` and depends on the operating point
    only through which units are on.
    """
    return output_weight @ (active.unsqueeze(-1) * hidden_weight)


def measure(
    body: CellBody,
    biases: CellBiases,
    *,
    ticks: int = DEFAULT_TICKS,
    burn_in: int = DEFAULT_BURN_IN,
    drive_correlation: float = DEFAULT_DRIVE_CORRELATION,
    drive_scale: float = DEFAULT_DRIVE_SCALE,
    operator_scale: float = DEFAULT_OPERATOR_SCALE,
    generator: torch.Generator | None = None,
) -> Measurement:
    """Measure every candidate over a driven trajectory, in one batched run.

    The measured object is the **chart's own round trip**,
    `d chart_{t+1} / d chart_t` through `encode` then `K` — what has to fail to
    contract for a cell to hold state. `decode` reads the chart out onto the
    node stalk and is not on that loop, so it is neither differentiated nor read
    for a fold margin.

    **The recurrence is `K @ J_encode`** since #138, and at construction
    `K = a.I` exactly, so this measures a scaled version of what it measured
    before and selection stays well-defined. `operator_scale` is that `a`, and
    :func:`operator_scale_rule` is what chooses it.

    **The narrow caveat, and it is the honest one:** selection places `tau` at
    construction, and a *learned* `K` will move it. That drift is #143's.

    Not measured at a frozen chart and stalk: the operating point varies as it
    will at runtime, which is the second of the go/no-go's three requirements
    and is what makes the region dwell reported alongside `tau` meaningful.
    """
    shape = body.shape
    stalks = driven_trajectory(
        body,
        biases,
        ticks=ticks,
        burn_in=burn_in,
        drive_correlation=drive_correlation,
        drive_scale=drive_scale,
        generator=generator,
    )
    like = body.encode_hidden_weight
    kwargs = {"device": like.device, "dtype": like.dtype}
    chart = torch.zeros((biases.cells, shape.k), **kwargs)
    identity = torch.eye(shape.k, **kwargs).expand(biases.cells, shape.k, shape.k)
    product = identity.clone()
    logs = torch.zeros(biases.cells, **kwargs)
    floor = torch.tensor(-50.0, **kwargs)
    finite = torch.ones(biases.cells, dtype=torch.bool, device=like.device)
    rho: list[torch.Tensor] = []
    margins: list[torch.Tensor] = []
    crossings = torch.zeros(biases.cells, **kwargs)
    region: torch.Tensor | None = None

    with torch.no_grad():
        for t in range(burn_in + ticks):
            # Through the body's own forward path, not a copy of it: a rig that
            # re-implemented `encode` and `step` would keep measuring the old
            # body after one was swapped in under it.
            fused_pre, fused = body.encode_parts(chart, stalks[t], biases)
            # `K = a.I` at construction, so advancing the chart is a scalar
            # multiply and the whole of `a`'s effect on this measurement is
            # visible right here (#140). It is not a free scalar: it multiplies
            # every eigenvalue below, and therefore every placed `tau`.
            chart = operator_scale * fused
            encode_active = (fused_pre > 0).to(chart.dtype)
            # Overflow reads downstream as no unit active, which is a zero
            # Jacobian and the fastest candidate in the sweep. Recorded here,
            # while the two are still distinguishable.
            finite &= torch.isfinite(chart).all(dim=-1)

            if t < burn_in:
                continue

            # The chart's own round trip: `encode`'s Jacobian restricted to the
            # chart half of its input, then the cell's `K`. The node stalk half
            # is where this tick's evidence entered and is not part of the loop.
            through_encode = _map_jacobian(
                body.encode_hidden_weight, body.encode_output_weight, encode_active
            )[:, :, : shape.k]
            jacobian = operator_scale * through_encode
            rho.append(torch.linalg.eigvals(jacobian).abs().amax(dim=-1))
            # Read from `encode` alone: after #138 it is the body's only
            # nonlinearity, so it is the only map that has folds at all. Read
            # through the body's own `fold_margin` for the same reason the
            # forward path above is the body's own: the live read (ADR-0019)
            # calls that method too, and construction nominating what the run
            # then decides means one measurement, not two that resemble it.
            margins.append(body.fold_margin(fused_pre))

            # A cell's activation region is which units of the round trip are on.
            # The chart crossing a fold is that pattern changing.
            here = encode_active
            if region is not None:
                crossings += (here != region).any(dim=-1).to(crossings.dtype)
            region = here

            # Renormalised every tick: the raw product under- and overflows well
            # inside 64 ticks, and only its log rate is wanted.
            product = jacobian @ product
            norm = torch.linalg.matrix_norm(product, ord=2)
            degenerate = norm <= 0
            logs += torch.where(norm > 0, norm.clamp(min=1e-30).log(), floor)
            product = torch.where(
                degenerate.view(-1, 1, 1),
                identity,
                product / norm.clamp(min=1e-30).view(-1, 1, 1),
            )

    spectra = torch.stack(rho)
    tau = -1.0 / spectra.clamp(min=1e-12, max=_RHO_CEILING).log()
    quantiles = torch.tensor(TAU_QUANTILES, device=like.device, dtype=like.dtype)
    return Measurement(
        tau=torch.quantile(tau, quantiles, dim=0).T.contiguous(),
        dwell=ticks / (1.0 + crossings),
        contraction=logs / ticks,
        rho_median=spectra.median(dim=0).values,
        finite=finite,
        rho_max=spectra.amax(dim=0),
        expansive=(spectra >= 1.0).to(logs.dtype).mean(dim=0),
        margin=torch.stack(margins).median(dim=0).values,
        ticks=ticks,
    )


@dataclass(frozen=True)
class Sweep:
    """Candidate bias vectors and what the driven trajectory said about each.

    The candidates are *drawn*; what makes the construction a selection is that
    most of them are thrown away.
    """

    candidates: CellBiases
    measurement: Measurement

    def acceptance(self, bands: tuple[Band, ...]) -> tuple[float, ...]:
        """Acceptance rate per band: the **reachability arm** of the go/no-go.

        What fraction of drawn bias vectors land in each band of the target
        range. Spread itself is no longer a falsifier — under selection it is
        constructed rather than observed — but a body unable to *reach* a band
        at any sampling budget kills the mechanism exactly as a spike would
        have.
        """
        tau = self.measurement.effective_timescale
        return tuple(float(band.holds(tau).to(tau.dtype).mean()) for band in bands)

    def reachable(
        self, bands: tuple[Band, ...], safety_factor: float = DEFAULT_SAFETY_FACTOR
    ) -> tuple[float, ...]:
        """Acceptance rate per band counting only candidates that stay contained.

        Reachable *and* usable. Candidates in the slow tail are reachable long
        before they are usable, which is what the `lambda` cap is for, so the two
        rates are reported side by side rather than one standing for both.
        """
        tau = self.measurement.effective_timescale
        contained = self.measurement.contained(safety_factor)
        return tuple(
            float((band.holds(tau) & contained).to(tau.dtype).mean()) for band in bands
        )

    def tail(
        self,
        thresholds: tuple[float, ...],
        safety_factor: float = DEFAULT_SAFETY_FACTOR,
    ) -> tuple[tuple[float, float], ...]:
        """`(threshold, acceptance)` for `tau >= threshold`, contained candidates only.

        How far up the slow tail selection can reach, at what acceptance rate —
        the arm that says whether a *different* reading of the demo's longest
        horizon would still have been reachable.
        """
        tau = self.measurement.effective_timescale
        contained = self.measurement.contained(safety_factor)
        return tuple(
            (t, float(((tau >= t) & contained).to(tau.dtype).mean())) for t in thresholds
        )


def sweep(
    body: CellBody,
    *,
    draws: int = DEFAULT_DRAWS,
    ticks: int = DEFAULT_TICKS,
    burn_in: int = DEFAULT_BURN_IN,
    drive_correlation: float = DEFAULT_DRIVE_CORRELATION,
    bias_variance: float | None = None,
    operator_scale: float = DEFAULT_OPERATOR_SCALE,
    generator: torch.Generator | None = None,
) -> Sweep:
    """Draw `draws` candidate bias vectors and measure each on a driven trajectory."""
    kwargs = {} if bias_variance is None else {"bias_variance": bias_variance}
    candidates = CellBiases(body.shape, draws, generator=generator, **kwargs)
    return Sweep(
        candidates=candidates,
        measurement=measure(
            body,
            candidates,
            ticks=ticks,
            burn_in=burn_in,
            drive_correlation=drive_correlation,
            operator_scale=operator_scale,
            generator=generator,
        ),
    )


#: @type chosen
#: @flexibility unknown
#: @warrant here
#: How many values of `a` :func:`operator_scale_rule` tries across the band.
#: Log-spaced, because `a` multiplies a rate.
DEFAULT_SCALE_STEPS = 12


def operator_scale_rule(
    body: CellBody,
    *,
    target: TargetRange,
    rho_k: float = DEFAULT_RHO_K,
    steps: int = DEFAULT_SCALE_STEPS,
    draws: int = DEFAULT_DRAWS,
    ticks: int = DEFAULT_TICKS,
    burn_in: int = DEFAULT_BURN_IN,
    drive_correlation: float = DEFAULT_DRIVE_CORRELATION,
    safety_factor: float = DEFAULT_SAFETY_FACTOR,
    generator: torch.Generator | None = None,
) -> float:
    """`a`: **the largest value in the band for which `slow_cap` still admits the
    target `tau` band** (ticket #140).

    A number this run produces **per body**, exactly as :meth:`Sweep.slow_cap`
    already is — the module's existing habit rather than a new one. Read
    plainly: *take the longest memory that still demonstrably forgets.*

    **Why the timescale constraint is the binding one, and not transmission.**
    Nothing rides on transmission at initialisation — the standing diagnosis
    (#120) is that the untrained graph transmits ~1e-14 whatever `a` is —
    whereas the construction-time go/no-go must be **valid**, and `a` multiplies
    every cell's placed `tau`. So `a` is fixed against the thing that would
    otherwise be silently wrong.

    **The two faces guard opposite failures**, and it is easy to get backwards.
    Lower face (`a` too small): `rho -> 0`, the chart is wiped every tick and
    the cell collapses toward its bias. Upper face (`a` too large): the cell
    **never forgets** and stops settling at all. Not a zeroing — a
    never-letting-go.

    Searched by scanning the band downward from its ceiling and returning the
    first value that admits the target's slow end, rather than by bisection: the
    scan makes no monotonicity assumption it has not measured, and the whole
    scan is one construction-time afternoon of the same sweep the go/no-go runs.

    **When nothing admits the target it returns the ceiling, not the floor**,
    and the direction is the whole point. The failure that puts the rule here is
    that cells forget *too fast* for the target's slow end, and answering that
    complaint with the smallest `a` in the band would hand back the fastest
    cells available — the two faces run opposite ways, and picking the wrong one
    is silent. The ceiling is the band's best attempt at the target, and the
    go/no-go is what reports that the attempt fell short; this function does not
    get to hide a no-go by choosing a number.

    **Measured on the default dome (#157), and worth knowing before reading a
    number out of this:** `a` is *not* the binding constraint on the present
    body. Containment holds for every one of 2048 candidates at every `a` in
    the band, so the upper face never binds, and `slow_cap` simply rises with
    `a` — 0.90 ticks at `a = 0.5` to 2.50 ticks at `a = 1.0`. Both readings of
    the demo's horizons ask for a slow end of 14 ticks or more, so the target is
    unreachable by roughly 6x whatever `a` does, and this returns the ceiling.
    That is a pre-existing property of a body whose effective timescale sits
    around one tick (`01-cell-and-sheaf.md`), not something the conversion
    caused — and raising `a` to the ceiling is the direction that helps it.
    """
    if steps < 1:
        raise ValueError(f"the scan takes at least one step, got {steps}")
    if rho_k < 1.0:
        raise ValueError(f"the operator band needs rho_k >= 1, got {rho_k}")
    floor = 1.0 / rho_k
    for i in range(steps):
        # Log-spaced from the ceiling down, so the *largest* admissible value is
        # the one found first.
        scale = (
            1.0
            if steps == 1
            else float(math.exp(math.log(floor) * i / (steps - 1)))
        )
        drawn = sweep(
            body,
            draws=draws,
            ticks=ticks,
            burn_in=burn_in,
            drive_correlation=drive_correlation,
            operator_scale=scale,
            generator=generator,
        )
        if drawn.measurement.slow_cap(safety_factor) >= target.slowest:
            return scale
    return 1.0


@dataclass(frozen=True)
class Selection:
    """The kept set: one bias vector per predicting cell, placed in its level's band.

    A filled selection is indexed by :attr:`Dome.predicting`'s row order, so it
    drops straight into the body beside the dome that ordered it; an unfilled one
    is shorter, and :attr:`cells` is then the only correct way to say which cell
    each row belongs to. Nothing in it records a rate — the placement happened
    once, here, and leaves nothing for a running cell to consult (`ADR-0005`).
    """

    biases: CellBiases
    bands: tuple[Band, ...]
    cells: tuple[int, ...]
    """The dome cell id each kept bias vector was placed on, in row order.

    This is the mapping, and it is the one to read: :attr:`biases` matches
    :attr:`Dome.predicting`'s row order only when :attr:`filled`, since a band
    short of candidates leaves cells of that level without one.
    """

    levels: tuple[int, ...]
    """The level each of those cells sits at, in the same order."""

    measurement: Measurement
    """The sweep's measurement, restricted to the kept candidates."""

    shortfall: dict[int, int] = field(default_factory=dict)
    """Cells a band could not be filled for, by level. Empty when the dome is filled."""

    @property
    def filled(self) -> bool:
        return not self.shortfall


def _predicting_levels(dome: Dome) -> tuple[int, ...]:
    """The level of each predicting cell, in :attr:`Dome.predicting`'s row order.

    Reads the construction layout, which is what the layout is for: selection is
    a construction-time act and this is the last of them. Nothing at runtime
    reads a cell's index, and nothing at runtime reads a band either.
    """
    return tuple(dome.cells[cell_id].index.level for cell_id in dome.predicting)


def select(
    dome: Dome,
    sweep_result: Sweep,
    *,
    target: TargetRange,
    overlap: float = DEFAULT_OVERLAP,
    safety_factor: float = DEFAULT_SAFETY_FACTOR,
) -> Selection:
    """Keep a set covering the target band, assigned by level, and write it into the graph.

    The bands overlap, so a candidate is usually eligible for two levels. The
    **scarcest** band takes what it needs first, which spends the overlap on the
    thin end of the range rather than on whichever level was asked first.
    Candidates outside every band, or not contained, are discarded — which is
    what makes this a selection rather than a draw.

    A band that cannot be filled is recorded in :attr:`Selection.shortfall` and
    **not** substituted for. Filling it from a neighbour would manufacture the
    gradient the construction is supposed to place, and the whole point of the
    reachability arm is that failing to reach a band is a result.
    """
    by_level: dict[int, list[int]] = {}
    for cell_id, level in zip(dome.predicting, _predicting_levels(dome)):
        by_level.setdefault(level, []).append(cell_id)
    ordered = sorted(by_level)
    bands = target.bands(len(ordered), overlap=overlap)

    measurement = sweep_result.measurement
    tau = measurement.effective_timescale
    contained = measurement.contained(safety_factor)
    eligible = {
        band.level: band.holds(tau) & contained
        for band in bands
    }

    # One candidate is one cell: a bias vector kept twice would be two cells that
    # are the same cell, which is the one thing the per-cell surface exists to
    # prevent. `available` is what no band has taken yet.
    available = torch.ones_like(contained)
    kept: dict[int, torch.Tensor] = {}
    scarcest = sorted(
        zip(ordered, bands), key=lambda pair: int(eligible[pair[1].level].sum())
    )
    for level, band in scarcest:
        offered = torch.nonzero(eligible[band.level] & available, as_tuple=False).flatten()
        take = offered[: len(by_level[level])]
        available[take] = False
        kept[level] = take

    rows: list[int] = []
    row_cells: list[int] = []
    row_levels: list[int] = []
    shortfall: dict[int, int] = {}
    for level in ordered:
        chosen = kept[level].tolist()
        if len(chosen) < len(by_level[level]):
            shortfall[level] = len(by_level[level]) - len(chosen)
        rows.extend(chosen)
        row_cells.extend(by_level[level][: len(chosen)])
        row_levels.extend([level] * len(chosen))

    index = torch.tensor(rows, dtype=torch.long)
    return Selection(
        biases=sweep_result.candidates.subset(index),
        bands=bands,
        cells=tuple(row_cells),
        levels=tuple(row_levels),
        measurement=_restrict(measurement, index),
        shortfall=shortfall,
    )


def _restrict(measurement: Measurement, index: torch.Tensor) -> Measurement:
    """The same measurement over a subset of candidates."""
    return Measurement(
        tau=measurement.tau[index],
        dwell=measurement.dwell[index],
        contraction=measurement.contraction[index],
        rho_median=measurement.rho_median[index],
        finite=measurement.finite[index],
        rho_max=measurement.rho_max[index],
        expansive=measurement.expansive[index],
        margin=measurement.margin[index],
        ticks=measurement.ticks,
    )


@dataclass(frozen=True)
class FoldMarginCheck:
    """`02-tick-semantics.md`'s `gain_v x offset <` fold margin, per cell across the taper.

    Reconciliation leaves a **standing offset** on the reconciled component of a
    node stalk, and that offset shifts the cell's operating point — which is
    where its timescale comes from. The divisor is the offset, whatever caused
    it, and not the disagreement floor: the floor is one contributor and, at
    construction, not the dominant one (#160, ADR-0007 as amended).

    The bound is read with the **per-cell gain** in it, `gain_v x offset <
    margin_v`, because read without it the bound would be the same number at
    every cell. `gain_v = gamma / (g_v^2 . c_v)` since
    [#190](https://github.com/NGL321/patchworks/issues/190) — one formula against
    each cell's own gauge, uniform across the interior. **What this function
    still forms is the superseded `max(sum_e m_e, rho^2 deg(v))`**, and swapping
    it is [#195](https://github.com/NGL321/patchworks/issues/195)'s, which owns
    the re-run and reports what the new denominator permits. Nothing here reads
    the result as a gate, so the stale denominator mis-states a nomination
    rather than passing a bad build.

    **A nomination, not a verdict** (ADR-0019). What the run decides is measured
    dwell, with :class:`patchworks.tick.FoldRead`'s live margin-against-offset
    as the attribution. The depth claim this docstring used to carry — that the
    bound binds hardest at the apex because `sum_e m_e` falls with depth — is
    **struck** and not replaced: #190 made `gain_v` uniform across the interior,
    so it binds on each cell's own margin draw, which is partly a draw.

    The offset is not known before anything runs, so what this produces is the
    **product** `gamma x offset` each cell can carry. Divide by an offset to get
    a cap on `gamma`, which `gamma <= 1` then caps again.
    """

    product_cap: torch.Tensor
    """`[predicting cells]`: the largest `gamma x offset` this cell can carry."""

    cells: tuple[int, ...]
    """Cell ids, in :attr:`product_cap`'s row order."""

    levels: tuple[int, ...]
    """The level of each cell, in the same order."""

    apex_level: int

    @property
    def binding(self) -> int:
        """Row index of the tightest cell — the one that caps `gamma` globally."""
        return int(self.product_cap.argmin())

    @property
    def apex_binds(self) -> bool:
        """Whether the tightest cell is at the apex.

        Reported, not expected. `02` used to claim the apex binds hardest; #160
        struck the claim with #190's uniform interior gain under it, so this is
        now a fact about one body's draw and about nothing systematic.
        """
        return self.levels[self.binding] == self.apex_level

    @property
    def cap(self) -> float:
        """The global cap on `gamma x offset`, set by the tightest cell.

        If the apex fails the bound, `gamma` is capped globally by the tightest
        cell; paying that everywhere costs only reconciliation speed at the rim,
        which is the cheapest thing in the system to give up.
        """
        return float(self.product_cap.min())

    def gamma_cap(self, offset: float) -> float:
        """The cap on the global `gamma` at a stated standing offset, `gamma <= 1`.

        The argument was named `floor` while the record took the disagreement
        floor to be what the bound divides by. It is the **standing offset**
        (#160): the displacement itself, whose dominant contributor at
        construction is model error rather than any floor.
        """
        if offset <= 0:
            raise ValueError(f"a standing offset is positive, got {offset}")
        return min(1.0, self.cap / offset)

    def by_level(self) -> tuple[tuple[int, int, float, float], ...]:
        """`(level, cells, median cap, tightest cap)` down the taper.

        The tightest *cell* is partly a draw — a cell's fold margin is
        uncorrelated with everything else about it. There is **no systematic
        part left for the level medians to show**: #190's `gain_v` is uniform
        across the interior, and #160 struck the depth claim without replacing
        it, on #178's finding that the quantity wanders 3.8x with no trend. The
        levels are kept as a reporting axis, not as a shape the record predicts.
        """
        rows = []
        for level in sorted(set(self.levels)):
            here = self.product_cap[
                torch.tensor([lv == level for lv in self.levels])
            ]
            rows.append(
                (level, int(here.numel()), float(here.median()), float(here.min()))
            )
        return tuple(rows)


def fold_margin_check(
    dome: Dome,
    margins: torch.Tensor,
    cells: tuple[int, ...],
    *,
    map_norm_bound: float = MAP_NORM_BOUND,
) -> FoldMarginCheck:
    """Run the `gain_v x offset <` fold margin check per cell across the taper.

    `margins` is one measured fold margin per entry of `cells`, which are dome
    cell ids — the selected cells', so the check runs on the body the graph will
    actually have rather than on a body it might have had.

    **Demoted, not deleted** (#140). The margin it reads is now `encode`'s
    alone, and the product it caps no longer *bounds* `gamma`: that bound was
    never binding — `DEFAULT_GAMMA` is 1.0, the global ceiling `02` permits —
    and `02`'s stated reason for it, that a shifted operating point changes the
    cell's effective timescale, is the premise #138 retired when timescale moved
    into `K`. What survives is a **nomination** made before the run (#160,
    ADR-0019), and the number it produces is what #155's fold-margin
    precondition consumes.
    """
    if margins.shape != (len(cells),):
        raise ValueError(
            f"one fold margin per cell, got {tuple(margins.shape)} for {len(cells)} cells"
        )
    denominator = torch.tensor(
        [
            max(dome.stalk_sums[cell_id], map_norm_bound**2 * dome.degrees[cell_id])
            for cell_id in cells
        ],
        dtype=margins.dtype,
    )
    return FoldMarginCheck(
        product_cap=margins * denominator,
        cells=tuple(cells),
        levels=tuple(dome.cells[cell_id].index.level for cell_id in cells),
        apex_level=max(_predicting_levels(dome)),
    )


def _measured(m: Measurement, indent: str) -> list[str]:
    """The four measured quantities, as the go/no-go requires each to be read."""
    q = lambda x, p: float(torch.quantile(x, p))  # noqa: E731 - a local shorthand
    tau = m.effective_timescale
    return [
        f"{indent}effective timescale, as quantiles never moments: p05/med/p95 = "
        f"{q(tau, 0.05):.2f} {q(tau, 0.5):.2f} {q(tau, 0.95):.2f} ticks, "
        f"max {float(tau.max()):.2f}, a {q(tau, 0.95) / q(tau, 0.05):.1f}x span",
        f"{indent}region dwell: p05/med/p95 = {q(m.dwell, 0.05):.2f} "
        f"{q(m.dwell, 0.5):.2f} {q(m.dwell, 0.95):.2f} ticks",
        f"{indent}realised contraction lambda: p05/med/p95 = {q(m.contraction, 0.05):+.3f} "
        f"{q(m.contraction, 0.5):+.3f} {q(m.contraction, 0.95):+.3f}, "
        f"{int((m.contraction >= 0).sum())} divergent, "
        f"{int((~m.finite).sum())} off the reals",
        f"{indent}max rho < 1 on {100 * float((m.rho_max < 1).to(tau.dtype).mean()):.1f}% "
        "of them (the cheap sufficient check); expansive regions visited: "
        f"{float(m.expansive.mean()):.4f}",
        f"{indent}fold margin: p05/med/p95 = {q(m.margin, 0.05):.4f} "
        f"{q(m.margin, 0.5):.4f} {q(m.margin, 0.95):.4f}",
    ]


@dataclass(frozen=True)
class GoNoGo:
    """The whole construction run: what it reached, what it kept, and what it kills.

    Read as a **shape check**. The body's widths are fixed but nothing is trained
    and there is no message stream to drive a cell with, so the sweep runs
    plausible chart and stalk sequences rather than real ones. It establishes
    that the mechanism is available. It does not produce the body's number.
    """

    horizons: DemoHorizons
    target: TargetRange
    """The range the demo's perturbation horizons derive, as derived.

    The measured `lambda` cap (:attr:`slow_cap`) is reported against it rather
    than folded into it: what a body reached is a result, and a target narrowed
    onto the result cannot be missed.
    """

    sweep: Sweep
    selection: Selection
    margin_check: FoldMarginCheck
    margin_check_on_draws: bool
    """Whether the margin check ran on drawn candidates rather than selected cells.

    True only when selection could not fill the taper, which is a failure of the
    timescale arm and not of the margin check: the two are independent inside a
    fixed body, so the check still reports.
    """

    safety_factor: float
    slow_cap: float
    draws: int

    @property
    def acceptance(self) -> tuple[float, ...]:
        return self.sweep.acceptance(self.selection.bands)

    @property
    def usable(self) -> tuple[float, ...]:
        return self.sweep.reachable(self.selection.bands, self.safety_factor)

    @property
    def kill(self) -> bool:
        """True when no draw reached the slow band.

        `05-timescales.md`: *if no draw reaches the slow band, this mechanism is
        dead and the afternoon that established it was well spent.* It is
        reported, never worked around — a band that cannot be reached is not
        filled from its neighbour.
        """
        return self.usable[-1] == 0.0

    @property
    def holds_nothing(self) -> bool:
        """True when measured `lambda` caps `tau` at or below the target's fast end.

        Stronger than :attr:`kill` and reported with it: not only is the slow
        band out of reach, no band of the range is.
        """
        return self.slow_cap <= self.target.fastest

    @property
    def dwell_short(self) -> torch.Tensor:
        """`[cells]` bool: selected cells whose region dwell is short against their `tau`.

        Not a kill and not a pass. Where this is widespread the mechanism is not
        dead but it is not *this* mechanism either — the cell decays at an
        average rate by averaging over unrelated regions
        (`05-timescales.md`, *The precondition*).

        "Long against" needs a factor and the record states none, so this reuses
        :attr:`safety_factor` — the one factor the run already states — rather
        than inventing a second number for the same word.
        """
        m = self.selection.measurement
        return m.dwell < self.safety_factor * m.effective_timescale

    def report(self) -> str:
        """The construction diagnostics, measured from the run."""
        m = self.selection.measurement
        sm = self.sweep.measurement
        lines = ["Bias selection", "==============", ""]

        lines.append("the target range, derived from the demo's perturbation horizons")
        lines.append(
            f"  fastest perturbation {self.horizons.fastest:g} ticks, longest "
            f"{self.horizons.longest:g} ticks"
        )
        lines.append(
            f"  target {self.target.fastest:.1f}-{self.target.slowest:.1f} ticks, "
            f"a {self.target.ratio:.1f}x range over {len(self.selection.bands)} "
            "overlapping bands"
        )
        lines.append(
            f"  measured lambda caps tau at {self.slow_cap:.2f} ticks per body "
            f"(safety factor {self.safety_factor:g}), against a slow end of "
            f"{self.target.slowest:.1f}"
        )
        if self.holds_nothing:
            lines.append(
                "  the cap falls at or below the fast end, so no band of the range "
                "is placeable: this body holds no timescale the demo asks for"
            )
        lines.append("")

        lines.append(
            f"reachability, over {self.draws:,} drawn bias vectors "
            f"({sm.ticks} ticks each)"
        )
        header = ("band", "ticks", "cells", "acceptance", "usable", "kept")
        rows = [header]
        kept = {level: self.selection.levels.count(level) for level in set(self.selection.levels)}
        for band, accept, usable in zip(self.selection.bands, self.acceptance, self.usable):
            wanted = kept.get(band.level, 0) + self.selection.shortfall.get(band.level, 0)
            rows.append(
                (
                    f"L{band.level}",
                    f"{band.lo:.1f}-{band.hi:.1f}",
                    str(wanted),
                    f"{100 * accept:.2f}%",
                    f"{100 * usable:.2f}%",
                    str(kept.get(band.level, 0)),
                )
            )
        widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
        for i, row in enumerate(rows):
            lines.append("  " + "  ".join(v.ljust(w) for v, w in zip(row, widths)))
            if i == 0:
                lines.append("  " + "  ".join("-" * w for w in widths))
        lines.append("")

        lines.append("the slow tail, contained candidates only")
        lines.append(
            "  "
            + "  ".join(
                f"tau>={t:g}: {100 * a:.3f}%"
                for t, a in self.sweep.tail(
                    (3.0, 10.0, 30.0, 100.0, 300.0), self.safety_factor
                )
            )
        )
        lines.append("")

        lines.append(f"what the {self.draws:,} draws measured")
        lines.extend(_measured(sm, "  "))
        lines.append("")

        if self.selection.filled:
            lines.append(f"selected {self.selection.biases.cells} cells, every band filled")
        else:
            short = ", ".join(
                f"L{level} {n} short" for level, n in sorted(self.selection.shortfall.items())
            )
            lines.append(f"selected {self.selection.biases.cells} cells, and {short}")
        if self.selection.biases.cells:
            lines.extend(_measured(m, "  "))
            lines.append(
                f"  dwell short against tau on {int(self.dwell_short.sum())} of "
                f"{self.selection.biases.cells} selected cells"
            )
        lines.append("")

        check = self.margin_check
        lines.append("gamma x floor < fold margin, per cell across the taper")
        if self.margin_check_on_draws:
            lines.append(
                "  on the drawn candidates, since selection could not fill the taper: "
                "margin and decay rate are uncorrelated inside a fixed body, so the "
                "check is unaffected by which draws were kept"
            )
        lines.append(
            f"  tightest cell: L{check.levels[check.binding]}, gamma x floor <= "
            f"{check.cap:.4f}"
            + ("  (the apex, where the spec expects it)" if check.apex_binds else "")
        )
        for level, cells, median, tightest in check.by_level():
            lines.append(
                f"    L{level} ({cells:3d} cells): gamma x floor <= {median:.4f} "
                f"at the median cell, {tightest:.4f} at the tightest"
            )
        lines.append(
            f"  at a disagreement floor of 1, gamma <= {check.gamma_cap(1.0):.3f}; "
            "a failing apex caps gamma globally, which costs only reconciliation "
            "speed at the rim"
        )
        lines.append("")

        lines.append("the go/no-go, read as a shape check")
        if self.kill:
            lines.append(
                "  KILL: no drawn bias vector reached the slow band "
                f"({self.selection.bands[-1]}), at any sampling budget this rig can "
                "spend. ADR-0005's mechanism is not available in this body, and the "
                "afternoon that established it was well spent."
            )
            if self.holds_nothing:
                lines.append(
                    f"  Nor any other band: measured lambda caps tau at "
                    f"{self.slow_cap:.2f} ticks against a target that starts at "
                    f"{self.target.fastest:.1f}."
                )
            lines.append(
                "  Reported, not worked around. The lever that would reach the band "
                "is sigma_w^2, and `05-timescales.md` sets it for containment only "
                "-- asking it to buy spread is what put a material fraction of "
                "regions past rho = 1 in #27's sweep."
            )
        elif not self.selection.filled:
            lines.append(
                "  the slow band is reachable but the taper could not be filled from "
                f"{self.draws:,} draws: {self.selection.shortfall}. A larger budget is "
                "the remedy; nothing is substituted from a neighbouring band."
            )
        else:
            lines.append(
                "  GO: every band of the target range is reachable and filled, and the "
                "mechanism is available."
            )
        dwell, implied = float(sm.dwell.median()), float(sm.effective_timescale.median())
        if dwell < self.safety_factor * implied:
            lines.append(
                f"  The precondition fails, and separately: region dwell (median "
                f"{dwell:.2f} ticks) is not long against the tau it implies (median "
                f"{implied:.2f}), where long means the run's stated factor of "
                f"{self.safety_factor:g}. A cell decaying at an average rate over "
                "unrelated regions is a different mechanism from the one specified, "
                "so even a reachable band would not have been this one."
            )
        return "\n".join(lines)


def go_no_go(
    dome: Dome,
    body: CellBody,
    *,
    horizons: DemoHorizons,
    draws: int = DEFAULT_DRAWS,
    ticks: int = DEFAULT_TICKS,
    burn_in: int = DEFAULT_BURN_IN,
    drive_correlation: float = DEFAULT_DRIVE_CORRELATION,
    overlap: float = DEFAULT_OVERLAP,
    safety_factor: float = DEFAULT_SAFETY_FACTOR,
    operator_scale: float = DEFAULT_OPERATOR_SCALE,
    drawn: Sweep | None = None,
    generator: torch.Generator | None = None,
) -> GoNoGo:
    """The whole construction run: sweep, cap, band, select, and check the margin.

    One sweep answers all of it, which is the point — the reachability arm, the
    `lambda` cap, the selection and `02-tick-semantics.md`'s fold-margin check
    are the same afternoon. Pass `drawn` to put a second target range to a sweep
    already taken: the range is what differs between readings of the demo's
    horizons, and the measurement does not depend on it.
    """
    if drawn is None:
        drawn = sweep(
            body,
            draws=draws,
            ticks=ticks,
            burn_in=burn_in,
            drive_correlation=drive_correlation,
            operator_scale=operator_scale,
            generator=generator,
        )
    draws = drawn.candidates.cells
    # The cap is reported rather than applied to the range. Narrowing the target
    # onto what a body reached would redefine the demo's ask as whatever was
    # available; and it needs no applying, because a candidate slower than the
    # cap is by construction one that failed containment and is already refused
    # by :func:`select`.
    cap = drawn.measurement.slow_cap(safety_factor)
    target = horizons.target
    selection = select(
        dome, drawn, target=target, overlap=overlap, safety_factor=safety_factor
    )
    if selection.filled:
        margins, cells = selection.measurement.margin, selection.cells
    else:
        # Selection could not place a body on the taper, and the fold-margin
        # check still has to run — it is the same sweep and the same afternoon,
        # and `02-tick-semantics.md` needs its answer whatever the timescale arm
        # says. Inside a fixed body a cell's fold margin is uncorrelated with its
        # decay rate (#42: corr(log rho, log margin) = -0.006), so the draws
        # stand in for the cells selection could not fill without the check
        # inheriting anything from which of them were kept. Every predicting cell
        # is covered — the check binds hardest at the apex, so a run that left the
        # deep levels out would report a looser cap than the taper permits.
        stand_in = torch.arange(len(dome.predicting)) % drawn.measurement.margin.numel()
        margins, cells = drawn.measurement.margin[stand_in], dome.predicting
    return GoNoGo(
        horizons=horizons,
        target=target,
        sweep=drawn,
        selection=selection,
        margin_check=fold_margin_check(dome, margins, cells),
        margin_check_on_draws=not selection.filled,
        safety_factor=safety_factor,
        slow_cap=cap,
        draws=draws,
    )
