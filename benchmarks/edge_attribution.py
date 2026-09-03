"""ADR-0004's exclusion procedure, as a rig: what is left when four causes are booked (#363).

[#333](https://github.com/NGL321/patchworks/issues/333) is *persistent,
structured, irreducible disagreement on edges that survives ADR-0004's
three-cause exclusion procedure — `colspan(D)` first, then curvature,
self-intersection and the lag floor.* Its `@cutoff` stood at `uncut` for a
reason [#351](https://github.com/NGL321/patchworks/issues/351) got right on the
second look: **it is not the quantity that is missing, it is the procedure.**
Per-edge disagreement is computed every tick and
:meth:`patchworks.diagnostics.Diagnostics.edge_reading` hands it over. Raw
disagreement is not the failure. What is left after four causes have been
excluded is, and of the four, two had instruments, one had an instrument with
nothing attributing a residual to it, and one had nothing at all.

This is that procedure. It runs the four exclusions **in ADR-0004's own order**,
attributes a share of every edge's energy to each, and reads what is left
against a measured reference level rather than a chosen one.

    python benchmarks/edge_attribution.py read
    python benchmarks/edge_attribution.py read --dome small --learn 200 --no-file

## The waterfall

One per-edge Dirichlet energy goes in and five numbers come out, summing back to
it. The order is the ADR's and the order is load-bearing: each stage books what
it can explain and hands the remainder on.

1. **`colspan(D)` — read first, and it costs one projection.** Freezing `decode`
   ([ADR-0014](../docs/adr/0014-the-linear-readout-is-gauge-fixed.md)) confines
   every cell's prediction to `im(D)`, *the same* `k`-dimensional subspace in
   every cell. ADR-0004 names the signature exactly: this cause is *shared
   across cells and fixed at construction*, so it shows as **a common direction
   in the residual across unrelated edges**, and *the direction is
   `colspan(D)`, known before the graph runs.* :func:`gauge_share` reads it
   there — on the **residual**, pulled back through each end's own map into the
   stalk it would have to be corrected in — against the chance level `k/n`,
   because a direction with no relationship to a `k`-dimensional subspace of an
   `n`-dimensional stalk still puts `k/n` of itself in it. ADR-0004: *failing to
   rule it out means the gauge is wrong, which is ADR-0014's own pre-registered
   falsification and not a fact about any edge* — so a large share here is a
   finding about [#325](https://github.com/NGL321/patchworks/issues/325), not
   about #333, and the report says so in those terms.

2. **Self-intersection — read forwards, at construction.** #49's amendment: a
   stalk too narrow to embed the piece it carries gives two distinct situations
   the same coordinates. Its distinguishing test — widen the stalk — *is not
   available on a running system*, so the ADR reads this cause forwards instead:
   *an embedding is generic once the coordinate count exceeds twice the piece's
   box-counting dimension.* :func:`piece_dimensions` measures the dimension and
   :func:`embeds` applies the criterion. An edge that fails it has a booked
   cause that is not #333 and leaves the procedure with its whole remainder
   attributed here — the same rule
   `prototypes/halting-thresholds/rules.py` applies as its first gate.

3. **The lag floor — the quiescent hold.** ADR-0007's protocol, and ADR-0004's
   answer to *how long before "does not fall with learning" is a claim rather
   than an impatience*: **an intervention, not patience.** Hold the world still
   and sweep configurations; lag drains, curvature does not. `hold_still` is
   imported from `benchmarks/detectability.py` rather than rewritten, and what
   drains across the hold is this cause's share.

4. **Curvature — the instrument existed and nothing attributed to it.**
   `benchmarks/graph_transmission.py` computes balanced Forman curvature
   (Topping et al., ICLR 2022), which they *prove* identifies the negatively
   curved edges "responsible for the over-squashing issue". So the sign is the
   bar and zero is cited rather than invented: an edge at `Ric < 0` books its
   standing residual to curvature and leaves.

5. **The residue.** Positively curved, wide enough to embed its piece, on the
   gauge, and still disagreeing after the hold. That is #333's failure, and
   nothing else in `benchmarks/` computes it.

## The reference level, which is measured and not chosen

`08-the-acceptance-demo.md` refuses a threshold that "would be a number invented
before anything was trained", and #156 found three of its four register entries
needed no threshold at all. This one needs a level, and there is one to be had
rather than invented.

#156's prototype used `06-graph-topology.md`'s **topology-only baseline** as a
level under the residual — *a measured construction quantity, and it is how this
rule gets a reference level without inventing one* — and stood it in at `0.05`,
because no run had produced it. It also found the comparison is **not optional**:
*without it the hold cheerfully halts a converged, healthy cell.*
:func:`patchworks.diagnostics.topology_only_h1` is that baseline as a count of
directions; :func:`patchworks.diagnostics.topology_only_energy`, added by this
ticket, is the same construction read as the **energy** a per-edge residual is
actually on — the disagreement this configuration would carry if the maps were
generic and full rank at ADR-0010's gauge. A residual no larger than it is
disagreement the graph's own shape produces whatever any map does.

So the reading is a **self-ratio** — standing residue against the level its own
topology produces — and the bar is `1`, in the register that
[#325](https://github.com/NGL321/patchworks/issues/325)'s `conduction ratio >= 1`
already sits in. At 1 the residue is exactly what the graph's shape accounts for;
above it, four causes have been booked and the disagreement is still larger than
topology explains.

**Reported metrics**, which is the list `tools/cutoff_report.py` reads a
`<metric>` from:

| metric | what it is |
|---|---|
| `residue_over_topology` | the headline. Σ standing residue over Σ topology-only level, on the surviving edges, **minimised over the sweep** |
| `residue_over_topology_opening` | the same ratio before any learning — #156's trap 4, so a crossing can be checked against a baseline rather than asserted |
| `surviving_edges` | how many edges reached stage 5 |
| `gauge_share` | the fraction of total energy stage 1 booked, in excess of chance. #325's quantity, not #333's |

## The sweep, and why the minimum

ADR-0007's hold is *hold the world still and sweep configurations*, and the
sweep is not decoration: the static floor is a function of configuration, so a
residual seen at one arrangement of the world is a fact about that arrangement.
Every quantity above is therefore taken per configuration and the headline is
the **minimum** across them — the part standing at every configuration in the
sweep, which is the only part the word *persistent* covers. The spread is
printed beside it, because a minimum with no spread beside it is a number nobody
can tell is representative.

## Like every other script here, it asserts nothing

`benchmarks/run_reporting.py` states the rule. A crossing is a report and a
label, never a failure and never a non-zero exit. What stands in the suite is
`tests/test_edge_attribution.py`, which holds the pure halves — the projector,
the dimension estimator, the criterion and the waterfall — against cases whose
answers are known, and smoke-tests the run on the small dome.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from patchworks.diagnostics import topology_only_energy
from patchworks.restriction import pair_index

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.append(_HERE)
# The build, the training loop and the hold are imported rather than copied.
# `hold_still` in particular: ADR-0007's protocol is one thing and a second copy
# of it drifting from `detectability`'s would leave two rigs claiming to have
# held the world still in two different ways.
import untrained_fixed_point as ufp  # noqa: E402
from detectability import hold_still  # noqa: E402
from graph_transmission import balanced_forman  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from cutoff_report import report as report_cutoffs  # noqa: E402

#: Configurations in the sweep. ADR-0007's hold is *hold the world still and
#: sweep configurations*, so one is not a sweep and the headline is a minimum
#: over these. Five is set by what the minimum costs to be worth quoting rather
#: than by anything measured: each configuration pays one hold and one
#: eigendecomposition of the generic complex, ~20 s on the real dome.
CONFIGURATIONS = 5

#: Ticks the world is driven for at each configuration before the hold, so the
#: lag floor has something to be a lag behind. Below the hold's length on
#: purpose: what is being established is motion, not convergence.
DRIVE = 200

#: Ticks of held world at each configuration. `detectability.HOLD`'s length and
#: `untrained_fixed_point.sensitivity`'s, for the same reason and taken from
#: them rather than chosen again here.
HOLD = 400

#: Ticks of trajectory the piece's dimension is estimated over, sampled during
#: the drive segment. A correlation dimension needs enough pairs for a log-log
#: slope to mean anything and saturates well before the cloud does; 200 points
#: gives ~20 000 pairs a cell.
CLOUD = 200

#: Training ticks before the read. The same length `benchmarks/floor_split.py`
#: and the rest of #351's rigs use, so the readings sit beside theirs.
LEARN = 3000

#: The scaling band the correlation dimension's slope is fitted over, as
#: quantiles of the pairwise distances. The tails are where a correlation
#: integral is not a power law — below the smallest inter-point spacing it is
#: counting nothing, above the cloud's diameter it has saturated at 1 — and
#: cutting them is the estimator's standard form rather than a tuning knob.
SCALING_BAND = (0.05, 0.35)


def gauge_projector(body) -> torch.Tensor:
    """`[n, n]`: the orthogonal projector onto `colspan(D)`.

    ADR-0004's fourth cause, and the whole of what ruling it out costs. `D` is
    `decode_weight`, `[n, k]`, **frozen** — so this is a construction quantity,
    the same in every cell, and known before the graph runs.

    Through a QR of `D` rather than through `D (DᵀD)⁻¹ Dᵀ`: the normal-equation
    form squares `D`'s condition number for no reason, and `D` is a random
    fan-in-scaled draw whose columns are near-orthogonal but not orthogonal.
    Float64, because everything this projector feeds is read in double.
    """
    d = body.decode_weight.detach().to(torch.float64)
    q, _ = torch.linalg.qr(d)
    return q @ q.T


def chance_alignment(shape) -> float:
    """`k / n`: how much of a *random* direction lies in `colspan(D)`.

    The level the gauge share is read against, and the reason stage 1 cannot be
    read raw. `colspan(D)` is a `k`-dimensional subspace of an `n`-dimensional
    stalk, so a direction with no relationship to it whatsoever still puts `k/n`
    of its energy there in expectation — 0.375 at `n = 32, k = 12`. A rig
    reporting an unadjusted 0.39 as *39% of the residual is the gauge's* would
    be reporting chance as a finding.

    A ratio of two construction dimensions, so it is neither measured nor
    chosen: it is arithmetic on numbers `06-graph-topology.md` already fixed.
    """
    return shape.k / shape.n


def excess_over_chance(observed: float, chance: float) -> float:
    """`(observed − chance) / (1 − chance)`, clamped at zero.

    How much of a fraction is there beyond what a direction with no relationship
    to the subspace would put there. `0` means *at chance*, `1` means *all of
    it*. Split out from :func:`gauge_reading` because it is the arithmetic that
    makes stage 1 a reading rather than a restatement of `k/n`, and it is worth
    a test of its own.
    """
    if chance >= 1.0:
        return 0.0
    return max(0.0, (observed - chance) / (1.0 - chance))


def gauge_reading(sheaf, projector: torch.Tensor):
    """`([edges], [edges])`: how much of each edge's residual the gauge forbids fixing.

    ADR-0004 names this cause's signature exactly: it is *shared across cells and
    fixed at construction*, so it shows as **a common direction in the residual
    across unrelated edges** — where curvature and self-intersection are per-edge
    and the lag floor is per-level — and *the direction is `colspan(D)`, known
    before the graph runs*.

    **The reading is on the residual, and the unfixable half is the one
    *outside* the gauge.** For each predicting end of each edge, `Fᵀr` is the
    residual `r = F_u x_u − F_v x_v` expressed as a direction in that cell's own
    `n`-dimensional stalk: the direction the cell would have to move its stalk
    along to reduce this edge's disagreement. Freezing `decode` confines the
    cell's stalk to `im(D)`, so **it can only move within `colspan(D)`** — and
    the component of `Fᵀr` orthogonal to `colspan(D)` is a correction the gauge
    forbids. That is ADR-0004's fourth cause in its own words: *wherever a
    stalk's real content lies outside it, the residue is persistent, structured,
    and arrives at the edge indistinguishable from curvature.*

    Boundary ends are skipped: a boundary stalk is the world's and `decode` never
    touched it (ADR-0006), so it has no gauge to be off.

    **Read against chance, not against zero.** A direction with no relationship
    to `colspan(D)` at all still puts `1 − k/n` of itself outside it — 0.625 at
    `n = 32, k = 12` — so an unadjusted 0.6 would be chance reported as a
    finding. Returns `(inside, share)`: the raw fraction lying *in* `colspan(D)`,
    which is what :func:`read` prints beside `k/n` so the ruling-out is visible
    rather than implied, and the excess-over-chance share stage 1 books.

    Both are fractions in `[0, 1]` by construction, being a projection of a
    vector into orthogonal complements. Nothing here can raise an edge's energy,
    which the first shape of this stage — projecting the stalks and re-reading
    the energy — could and did, on 52 of 54 edges: taking content off one end of
    an edge and not the other moves the two ends apart, so it measured the
    projection rather than the gauge.
    """
    dome = sheaf.dome
    n = dome.shape.n
    chance = 1.0 - chance_alignment(dome.shape)
    predicting = set(dome.predicting)
    with torch.no_grad():
        maps = sheaf.maps
        outgoing = maps.restrict(sheaf.stalks[sheaf.layout.pair_positions])
        ends = outgoing.reshape(-1, 2, maps.edge_width).double()
        residual = ends[:, 0] - ends[:, 1]
        weights = maps.maps.detach().double()
    inside = torch.zeros(len(dome.edges), dtype=torch.float64)
    share = torch.zeros(len(dome.edges), dtype=torch.float64)
    for edge in dome.edges:
        held = within = 0.0
        for side, cell_id in enumerate((edge.u, edge.v)):
            if cell_id not in predicting:
                continue
            pulled = weights[pair_index(edge.id, side), : edge.m, :n].T @ residual[
                edge.id, : edge.m
            ]
            held += float(pulled.pow(2).sum())
            within += float((projector @ pulled).pow(2).sum())
        if held <= 0.0:
            continue
        inside[edge.id] = within / held
        share[edge.id] = excess_over_chance(1.0 - within / held, chance)
    return inside, share


def per_edge_energy(sheaf, stalks: torch.Tensor) -> torch.Tensor:
    """`[edges]`: `‖F_u x_u − F_v x_v‖²` on the stalk buffer handed in.

    :meth:`patchworks.diagnostics.Diagnostics.edge_reading` computes this on the
    sheaf's own stalks and hands it over paired with effective rank, which is
    the right shape for the instrument it belongs to and the wrong one here:
    this rig needs the *same* quantity on a **counterfactual** configuration —
    the gauge-projected one — and the pairing rule exists to stop a caller
    reading half an instrument, not to stop one asking what the energy would be
    somewhere else. Nothing here is a diagnostic reading; it is arithmetic on a
    hypothetical, and it is never recorded as a reading.
    """
    with torch.no_grad():
        outgoing = sheaf.maps.restrict(stalks[sheaf.layout.pair_positions])
        ends = outgoing.reshape(-1, 2, sheaf.maps.edge_width)
        return (ends[:, 0] - ends[:, 1]).pow(2).sum(-1).double()


def correlation_dimension(cloud: np.ndarray, band=SCALING_BAND) -> float:
    """Grassberger–Procaccia correlation dimension of a point cloud.

    The slope of `log C(r)` against `log r`, with `C(r)` the fraction of point
    pairs closer than `r`, fitted over the middle of the distance distribution.

    **This is a lower bound on the box-counting dimension the criterion is
    stated in, and the direction of the bound matters.** Sauer, Yorke &
    Casdagli's theorem — which is what ADR-0004 and ADR-0007 both cite — is
    stated with the box-counting dimension; the correlation dimension is the
    estimator the same literature actually uses, and it never exceeds the
    box-counting dimension. So `m > 2 · d_corr` is **easier** to satisfy than
    the real criterion, and this stage therefore *under*-books
    self-intersection: every edge it excludes on this ground genuinely fails,
    and some that pass may not. That is the conservative direction for a
    procedure whose output is a residue — an under-booked exclusion leaves work
    in the residue rather than taking it out — and it is stated here rather than
    left for a reader to derive.

    Returns `0.0` for a cloud with no spread, which is the honest answer: a cell
    whose stalk never moves carries a piece of dimension zero and no stalk width
    is too narrow for it.
    """
    points = np.asarray(cloud, dtype=np.float64)
    if points.shape[0] < 8:
        return 0.0
    difference = points[:, None, :] - points[None, :, :]
    distances = np.sqrt((difference**2).sum(-1))
    upper = distances[np.triu_indices(points.shape[0], k=1)]
    upper = upper[upper > 0.0]
    if upper.size < 8:
        return 0.0
    low, high = np.quantile(upper, band)
    if not (high > low > 0.0):
        return 0.0
    radii = np.exp(np.linspace(np.log(low), np.log(high), 12))
    counts = np.array([(upper < r).mean() for r in radii])
    keep = counts > 0.0
    if keep.sum() < 3:
        return 0.0
    slope = np.polyfit(np.log(radii[keep]), np.log(counts[keep]), 1)[0]
    return float(max(0.0, slope))


def embeds(width: int, dimension: float) -> bool:
    """#49's criterion: `width > 2 · dimension`, read forwards.

    *An embedding is generic once the coordinate count exceeds twice the piece's
    box-counting dimension* (ADR-0004, from Sauer, Yorke & Casdagli). Strict,
    because the theorem's guarantee is genericity above the bound and says
    nothing at it.
    """
    return width > 2.0 * dimension


@dataclass(frozen=True)
class Attribution:
    """One configuration's waterfall: five per-edge shares that sum back to the total.

    Every field is `[edges]` and in float64.
    The identity is exact on every edge and in two halves:
    `total = gauge + self_intersection + lag + standing`, and
    `standing = curvature + reference + residue`.
    `tests/test_edge_attribution.py` holds both.
    """

    total: np.ndarray
    """The per-edge Dirichlet energy the waterfall starts from, driven."""

    gauge: np.ndarray
    """Stage 1: the share whose direction is `colspan(D)`, in excess of chance."""

    self_intersection: np.ndarray
    """Stage 2: the whole remainder, on edges too narrow to embed their piece."""

    lag: np.ndarray
    """Stage 3: what drained across the quiescent hold."""

    curvature: np.ndarray
    """Stage 4: the standing remainder, on edges at `Ric < 0`."""

    reference: np.ndarray
    """The share of the standing residual the topology-only level accounts for —
    **not a cause**, a reference. On a surviving edge it is the level itself and
    the residue is what stands above it; on an edge standing below the level it
    is the whole standing residual, which is what *this edge is disagreeing by
    no more than its own shape produces* looks like as a share."""

    residue: np.ndarray
    """Stage 5: what is left. #333's failure, per edge."""

    surviving: np.ndarray
    """`[edges]` bool: reached stage 5 booked to nothing, and standing above the
    reference level once there."""

    standing: np.ndarray
    """What was left after stages 1–4, before the reference level was held out.
    Kept so :func:`clamps` and the report can say what the comparison was
    against rather than only what survived it."""

    grew: np.ndarray
    """`[edges]` bool: the hold left this edge disagreeing *more* than under
    drive, so stage 3's share is a clamp rather than a measurement. Carried
    rather than swallowed — see :func:`attribute`."""

    alignment: np.ndarray
    """`[edges]`: the **raw** fraction of the pulled-back residual lying in
    `colspan(D)`, before chance was taken off. Carried so :func:`read` can print
    it beside `k/n`: stage 1's job is to *rule the gauge out*, and a stage that
    books nothing is only legible as a ruling-out if the reader can see the
    number it was compared against."""

    def ratio(self) -> float:
        """The headline at this configuration: residue against reference, over the survivors.

        Summed over the surviving edges rather than averaged per edge, so one
        edge carrying a large residue is not diluted by a thousand carrying
        none — the failure is *per edge* and ADR-0004 says so.

        Zero when nothing survived, which is the honest reading and not a
        missing one: every edge's disagreement was booked to a named cause, and
        that is what #333 not happening looks like.
        """
        keep = self.surviving
        if not keep.any():
            return 0.0
        below = float(self.reference[keep].sum())
        if below <= 0.0:
            return 0.0
        return float(self.residue[keep].sum() / below)


def attribute(
    total: np.ndarray,
    gauge: np.ndarray,
    held: np.ndarray,
    curvature: np.ndarray,
    reference: np.ndarray,
    embedded: np.ndarray,
    *,
    alignment: np.ndarray | None = None,
) -> Attribution:
    """Run the waterfall. Pure arithmetic on six `[edges]` arrays, and the seam the tests hold.

    `total` is the driven per-edge energy; `gauge` is :func:`gauge_share`'s
    excess-over-chance fraction; `held` is the driven-scale energy re-read at the
    end of the quiescent hold, already net of its own gauge share; `curvature` is
    balanced Forman curvature; `reference` is the topology-only level;
    `embedded` is #49's criterion, per edge.

    **One clamp, and it is reported rather than swallowed.** A hold can leave an
    edge disagreeing *more* than it did under drive — nothing in ADR-0007
    promises otherwise, and on a graph that has not converged the sheaf goes on
    moving after the world stops. On such an edge the lag share would be
    negative, which is not a share, so it books zero and the edge is flagged in
    :attr:`Attribution.grew`. Booking zero rather than a negative number is the
    conservative direction: the whole driven remainder is passed on as standing,
    so nothing is taken *out* of the residue by a stage that could not measure
    itself. :func:`clamps` prints how often it fired, because a stage that books
    nothing and a stage that could not be computed must not read the same.

    Stage 1 needs no clamp: :func:`gauge_share` is a fraction in `[0, 1]` by
    construction, being a projection of a vector into orthogonal complements.
    """
    gauge_energy = gauge * total
    remaining = total - gauge_energy

    self_intersection = np.where(embedded, 0.0, remaining)
    live = embedded.copy()
    remaining = np.where(live, remaining, 0.0)

    # What drained across the hold is lag; what stood is not. `held` is already
    # net of its own gauge share, so this compares like with like -- reading the
    # drain against the unbooked total would book stage 1's share twice.
    grew = held > remaining
    standing = np.where(live, np.minimum(remaining, np.maximum(0.0, held)), 0.0)
    lag = np.where(live, remaining - standing, 0.0)

    # `standing` is **not** zeroed on the edges curvature books, because it is
    # what the second identity is stated against: `curvature + reference +
    # residue == standing` on every edge, with curvature taking the whole of it
    # where it fires. Zeroing it here would leave the decomposition summing to
    # less than it decomposed on exactly the edges a cause had been found for.
    negative = curvature < 0.0
    curvature_share = np.where(live & negative, standing, 0.0)
    live = live & ~negative

    # An edge whose standing residual the topology-only level already accounts
    # for has not survived the procedure: it is disagreeing by no more than a
    # graph of this shape with generic full-rank maps would, which is the
    # comparison `06-graph-topology.md` requires and the one #156 found is not
    # optional -- *without it the hold cheerfully halts a converged, healthy
    # cell.*
    surviving = live & (standing > reference)
    accounted = np.where(live, np.minimum(standing, reference), 0.0)
    residue = np.where(surviving, standing - reference, 0.0)
    return Attribution(
        total=total,
        gauge=gauge_energy,
        self_intersection=self_intersection,
        lag=lag,
        curvature=curvature_share,
        reference=accounted,
        residue=residue,
        surviving=surviving,
        standing=standing,
        grew=grew & live,
        alignment=np.zeros_like(total) if alignment is None else alignment,
    )


def piece_dimensions(sheaf, cloud: np.ndarray) -> np.ndarray:
    """`[predicting cells]`: each cell's piece's dimension, off its node stalk trajectory.

    The cloud is `[ticks, predicting cells, n]` — the cell's own public face
    over the drive segment, in the `n`-dimensional space the piece is described
    in. Measuring it there and not in the edge stalk is what keeps the criterion
    from being circular: a cloud already compressed into `m` coordinates cannot
    have a measured dimension above `m`, so asking whether `m` is wide enough
    from inside it always says yes.
    """
    return np.array(
        [correlation_dimension(cloud[:, i, :]) for i in range(cloud.shape[1])]
    )


def edge_dimensions(dome, per_cell: np.ndarray) -> np.ndarray:
    """`[edges]`: the dimension of the piece each edge has to carry.

    The **smaller** of its two predicting ends', because the structure two cells
    share cannot be richer than the poorer of the two pieces it is shared
    between, and #49's criterion is about the piece the *stalk* carries. A
    boundary end contributes nothing: it holds no chart and owns no piece
    (ADR-0006), so a boundary-incident edge is sized by its predicting end
    alone.
    """
    row = {cell_id: i for i, cell_id in enumerate(dome.predicting)}
    out = np.zeros(len(dome.edges))
    for edge in dome.edges:
        ends = [per_cell[row[c]] for c in (edge.u, edge.v) if c in row]
        out[edge.id] = min(ends) if ends else 0.0
    return out


def sample(agent, ticks: int, keep: int, seed: int):
    """Drive the world for real, keeping the last `keep` ticks of predicting node stalks.

    Through :func:`patchworks.agent.run` and not through `hold_still`: this is
    the segment the *lag* floor is supposed to accumulate over, and a lag floor
    is a function of motion. Running the drive against a held world would leave
    stage 3 with nothing to drain and quietly hand the lag share to curvature,
    which is the one confusion ADR-0007's hold exists to prevent.

    Returns the cloud and the last tick's outcome, whose observation and applied
    torque are what the hold then holds the world at.
    """
    positions = agent.sheaf.layout.predicting_positions
    trail: list[np.ndarray] = []
    outcome = None
    for step, outcome in enumerate(ufp.run(agent, ticks, seed=seed)):
        if step >= ticks - keep:
            with torch.no_grad():
                trail.append(agent.sheaf.stalks[positions].detach().clone().numpy())
    cloud = np.stack(trail) if trail else np.zeros((0, positions.shape[0], 1))
    return cloud, outcome


def configuration(
    agent, seed: int, drive: int, hold: int, projector, curvature, keep: int
) -> Attribution:
    """One arm of the sweep: rearrange the world, drive it, hold it still, attribute.

    The order is ADR-0007's protocol and nothing here reorders it. The drive
    segment is what gives the lag floor something to be a lag behind; the hold
    is what drains it. Both readings are taken on the *same* graph with the
    *same* maps — no learning runs in either — so the only thing that differs
    between them is whether the world was moving, which is the whole of what
    the hold is for.

    The gauge share is taken **twice**, once under drive and once at the end of
    the hold, and each energy is netted of its own. Reusing the driven share on
    the held reading would compare a residual to a projection of a different
    configuration, and the two stages would leak into each other.
    """
    cloud, outcome = sample(agent, drive, keep, seed)
    sheaf = agent.sheaf
    total = per_edge_energy(sheaf, sheaf.stalks).numpy()
    alignment, gauge = (t.numpy() for t in gauge_reading(sheaf, projector))

    hold_still(agent, outcome.observation, outcome.applied, hold)
    held_total = per_edge_energy(sheaf, sheaf.stalks).numpy()
    held = held_total * (1.0 - gauge_reading(sheaf, projector)[1].numpy())
    reference = topology_only_energy(sheaf).numpy()

    widths = np.array([edge.m for edge in sheaf.dome.edges], dtype=float)
    dimensions = edge_dimensions(sheaf.dome, piece_dimensions(sheaf, cloud))
    embedded = np.array(
        [embeds(w, d) for w, d in zip(widths, dimensions)], dtype=bool
    )
    return attribute(
        total, gauge, held, curvature, reference, embedded, alignment=alignment
    )


def readings(arms: list[Attribution]) -> dict[str, float]:
    """What `tools/cutoff_report.py` reads a `<metric>` from.

    The headline is the **minimum** across the sweep, for the reason this
    module's docstring gives: the part standing at every configuration is the
    only part *persistent* covers.
    """
    ratios = [arm.ratio() for arm in arms]
    return {
        "residue_over_topology": float(min(ratios)) if ratios else 0.0,
        "surviving_edges": float(min(int(a.surviving.sum()) for a in arms))
        if arms
        else 0.0,
        "gauge_share": float(
            min(
                (a.gauge.sum() / a.total.sum()) if a.total.sum() > 0 else 0.0
                for a in arms
            )
        )
        if arms
        else 0.0,
    }


def show(arms: list[Attribution], label: str, chance: float) -> None:
    """The waterfall, as a table, with the sweep's spread beside every number."""
    print(f"\n### the waterfall, {label} — {len(arms)} configurations\n")
    total = np.array([a.total.sum() for a in arms])
    print(f"  {'stage':>22} {'share of total':>16} {'across the sweep':>26}")
    for name, values in (
        ("1. colspan(D)", [a.gauge.sum() for a in arms]),
        ("2. self-intersection", [a.self_intersection.sum() for a in arms]),
        ("3. lag floor", [a.lag.sum() for a in arms]),
        ("4. curvature", [a.curvature.sum() for a in arms]),
        ("   topology-only level", [a.reference.sum() for a in arms]),
        ("5. residue", [a.residue.sum() for a in arms]),
    ):
        shares = np.array(values) / np.where(total > 0, total, 1.0)
        print(
            f"  {name:>22} {shares.mean():15.4f}  "
            f"{shares.min():11.4f} to {shares.max():.4f}"
        )
    alive = [a.alignment > 0 for a in arms]
    aligned = [float(a.alignment[k].mean()) for a, k in zip(arms, alive)]
    print()
    print(
        f"  stage 1's raw reading: {min(aligned):.4f} to {max(aligned):.4f} of the "
        f"pulled-back residual lies in colspan(D), against a chance level of "
        f"{chance:.4f}"
    )
    counts = [int(a.surviving.sum()) for a in arms]
    edges = len(arms[0].total)
    print(
        f"\n  edges reaching stage 5: {min(counts)} to {max(counts)} of {edges}"
    )
    ratios = [a.ratio() for a in arms]
    print(
        f"  residue / topology-only level: {min(ratios):.4g} to {max(ratios):.4g}"
        f"   (the headline is the minimum)"
    )


def clamps(arms: list[Attribution]) -> None:
    """How often :func:`attribute`'s one clamp fired, across the sweep.

    Printed rather than swallowed. A hold that leaves most edges disagreeing
    more than they did under drive is a hold being read on a graph that has not
    settled, and stage 3 booked zero there because it could not measure itself
    — not because there was no lag. A decomposition that hid that inside a `max`
    would let a reader take an empty lag row for a finding.
    """
    grew = [int(a.grew.sum()) for a in arms]
    live = [int((a.standing > 0).sum()) for a in arms]
    edges = len(arms[0].total)
    print()
    print(
        f"  the hold left {min(grew)} to {max(grew)} edges of {edges} "
        f"disagreeing more than under drive, so stage 3 booked nothing there; "
        f"{min(live)} to {max(live)} edges reached stage 4 still standing"
    )


def read(
    name: str,
    split: str,
    seed: int,
    learn: int,
    arms: int,
    drive: int,
    hold: int,
    keep: int,
    file_cutoffs: bool = True,
) -> None:
    """The whole read: sweep once untrained, train, sweep again, file the verdict.

    **Twice, and the first sweep is #156's trap 4.** *A bar that fires on the
    architecture working* is the trap that prototype hit and recorded, and the
    answer #351 used on #324 is to make the rig carry its own baseline, because
    "a verdict whose baseline lives in someone's shell history is a verdict
    nobody can check". #333's failure is disagreement that **does not fall with
    learning**, so the opening sweep is not decoration: it is the only thing
    that distinguishes a residue learning never touched from a residue that is
    simply what an untrained graph looks like.
    """
    _env, agent = ufp.build(name, split, seed)
    dome = agent.dome
    print(f"dome {name}: {len(dome.cells)} cells, {len(dome.edges)} edges")
    projector = gauge_projector(agent.sheaf.body)
    print(
        f"colspan(D): rank {int(round(float(projector.trace())))} of "
        f"n = {dome.shape.n}, frozen at construction"
    )
    curvature = balanced_forman(dome)
    print(
        f"balanced Forman curvature: {int((curvature < 0).sum())} of "
        f"{len(dome.edges)} edges negatively curved"
    )

    def sweep(label: str) -> list[Attribution]:
        out = []
        for i in range(arms):
            out.append(
                configuration(
                    agent, seed * 1000 + i, drive, hold, projector, curvature, keep
                )
            )
            print(
                f"  {label} configuration {i + 1}/{arms}: "
                f"{int(out[-1].surviving.sum())} edges to stage 5, "
                f"ratio {out[-1].ratio():.4g}",
                flush=True,
            )
        return out

    print(f"\nopening sweep, before any learning ({arms} configurations)...", flush=True)
    opening = sweep("opening")

    print(f"\ntraining {learn} ticks with both rules...", flush=True)
    if learn:
        ufp.taught(agent, learn, seed)
    trained = sweep("trained")

    show(opening, "before learning", chance_alignment(dome.shape))
    show(trained, "after learning", chance_alignment(dome.shape))
    clamps(trained)

    measured = readings(trained)
    measured["residue_over_topology_opening"] = readings(opening)[
        "residue_over_topology"
    ]
    print("\n### what the register reads\n")
    for key in sorted(measured):
        print(f"  {key:>34} {measured[key]:.6g}")
    # The measurement half of the cutoff mechanism (#284). #333 cuts on this
    # rig. It states a verdict, files the run, and asserts nothing.
    report_cutoffs("edge_attribution", measured, file=file_cutoffs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="measurement", required=True)
    reading = sub.add_parser("read", help="the exclusion procedure, end to end")
    reading.add_argument("--dome", default="full", choices=("small", "full"))
    reading.add_argument("--split", default="train")
    reading.add_argument("--seed", type=int, default=0)
    reading.add_argument("--learn", type=int, default=LEARN)
    reading.add_argument("--configurations", type=int, default=CONFIGURATIONS)
    reading.add_argument("--drive", type=int, default=DRIVE)
    reading.add_argument("--hold", type=int, default=HOLD)
    reading.add_argument("--cloud", type=int, default=CLOUD)
    reading.add_argument(
        "--no-file",
        dest="file_cutoffs",
        action="store_false",
        help="print the verdict and touch the tracker not at all",
    )
    arguments = parser.parse_args(argv)
    read(
        arguments.dome,
        arguments.split,
        arguments.seed,
        arguments.learn,
        arguments.configurations,
        arguments.drive,
        arguments.hold,
        arguments.cloud,
        file_cutoffs=arguments.file_cutoffs,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
