"""The diagnostics that run on a cadence, and the one baseline construction owes.

`docs/spec/01-cell-and-sheaf.md`, *Known exposure* (over-smoothing) and *`H⁰` is
the private features*, together with `docs/spec/06-graph-topology.md`, *What the
cycles do*, are the whole of what this module implements. ADR-0010 fixes the
shape of the instrument and ADR-0007 fixes the contrast it is read against.

**Nothing here is part of the architecture.** These are measurements, on exactly
the footing `docs/spec/03-the-sandbox.md` gives `info` and
`docs/spec/10-the-demo-surface.md` gives the demo surface: for looking at, never
fed back. No cell reads anything computed here, nothing here writes a chart, a
node stalk, an edge buffer or a map, and switching the whole module off changes
no trajectory — `tests/test_diagnostics.py` asserts that bit for bit rather than
leaving it to inspection.

That footing is load-bearing for one reading in particular.
:meth:`Diagnostics.whole_graph` assembles the sheaf's coboundary over the entire
graph and decomposes it, which is precisely the **global aggregate**
graph-locality exists to forbid (ADR-0002). It is legitimate here for the one
reason it is legitimate of `info`: it is read from outside by an experimenter,
never by a cell, and no quantity it produces re-enters the graph.

## The instrument is a pair, and the pairing is enforced here

Over-smoothing in this architecture is **the error signal vanishing, not a
quality loss**. Disagreement *is* the Dirichlet energy and is the only edge-level
error signal there is, so the classical failure mode arrives as the
disappearance of the quantity both halves of the local learning rule are
computed from — and of the only instrument that would show it happening.

Falling energy is not on its own a pathology. Boundary cells are written by the
world every tick, prediction pushes states off the consistent set continuously,
and `H⁰` is large by construction, so a consistent section here is content-rich
rather than degenerate. Energy draining **at rest** is the quiescent hold
(ADR-0007) working as designed. The alarming case is energy falling **while the
world drives**, and the two are told apart by reading per-edge energy alongside
per-edge **effective rank**:

* energy down under drive, effective rank sliding toward 1 across the fleet, is
  parameter collapse;
* energy down at rest, effective rank steady, is the lag floor draining.

Neither reading separates them alone, so this module offers no way to obtain one
without the other. The pairing is in the API rather than in this paragraph:
:class:`EdgeReading` is the only thing that carries either quantity, it requires
both, and it checks that the two describe the same edges. There is no public
function here that returns a per-edge energy on its own or a per-edge effective
rank on its own, and there is no way to record one without the other.

**No rank floor is imposed, here or anywhere.** Learned rank-deficiency is
wanted — it is the mechanism `06-graph-topology.md` relies on to enlarge `H⁰`
through a functionally dead but structurally present edge — and its degenerate
limit is instrumented rather than forbidden. Nothing in this module clamps,
warns on, refuses or floors an effective rank: a fleet of rank-1 maps is
reported as a fleet of rank-1 maps.

That holds in the arithmetic and not only in the intent. The participation
ratio is taken on the **unit-normalised** map, where it needs no epsilon
anywhere — `1 / ‖G Gᵀ‖_F²` for `G = F/‖F‖_F`, a denominator that lies in
`[1/m, 1]` for any `G` at all. A stabilising constant added to the denominator
of `‖F‖_F⁴ / ‖F Fᵀ‖_F²` would be worse than untidy: the ratio is exactly
scale-invariant, so an *absolute* epsilon reads as a rank that falls with the
norm, and a healthy fleet at a small norm would report collapse. That is the
false reading the pair exists to prevent, arriving through the instrument
itself, and it would fire exactly where ADR-0010 says a map's norm is not a
diagnostic. The only special case is `F = 0`, which reads `0` — *below* the
degenerate limit of 1, never clamped up to it.

## What the world was doing is declared, not measured

A reading is worthless without the condition it was taken under, so
:meth:`Diagnostics.observe` will not take one without being told. The two
conditions are ADR-0010's driven/quiescent contrast — `01-cell-and-sheaf.md`'s
"while the world drives" and "at rest" — and they are a fact about what the
**experimenter arranged**, not about anything the sheaf can see. ADR-0007's
quiescent hold is a protocol: hold the world still, sweep configurations, watch
what drains. The sandbox supports it without a special mode, which is exactly
why nothing in the environment announces it. So :class:`Condition` is a
declaration by whoever arranged the world, and inferring it instead would mean
inventing a threshold on "still enough" that no part of the record fixes.

Note the word: :attr:`Condition.DRIVEN` is *the world driving*, and has nothing
to do with the **drive** boundary cell, whose standing assertion is written
every tick under either condition (:mod:`patchworks.agent`). The record's own
pair of words for this contrast is driven/quiescent and that is the pair used
here.

## Two cadences, because the two halves cost three orders of magnitude apart

The per-edge pair is a batched matrix product over edge endpoints and costs
a few milliseconds on the real dome. The whole-graph reading is one symmetric
eigendecomposition of `δ δᵀ` and costs ~16 seconds there. One cadence for both
would either pay the second every time the first is wanted or starve the first.

So there are two, and :attr:`Diagnostics.whole_graph_every` must be a multiple of
:attr:`Diagnostics.every`. That is what keeps the record coherent: every
whole-graph reading lands on a tick that also carries its paired per-edge
reading, of the same configuration, under the same declared condition, so the
expensive numbers can always be lined up against the cheap ones rather than
interpolated between them.

One note on naming, because the mismatch is deliberate. The record's word for
this is **cadence** — ADR-0010 says "on the diagnostic cadence, not per tick" —
and the prose here says so throughout, but no *identifier* in this module does.
`tests/test_timescale.py` scans every module of the package for the vocabulary a
per-cell schedule would be written in, and `cadence` is one of the words it
refuses; it reads identifiers out of the syntax tree and never a string, which
is what lets this paragraph say the word while `every` and `whole_graph_every`
carry it in code. Staying inside the scan is worth more than the four
characters: this module is an instrument and could have been exempted wholesale,
and then nothing would be watching it for the other four words.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

from .graph import Dome
from .restriction import pair_index
from .tick import Sheaf

__all__ = [
    "BASELINE_SEED",
    "DEFAULT_EVERY",
    "DEFAULT_WHOLE_GRAPH_EVERY",
    "Condition",
    "Diagnostics",
    "EdgeReading",
    "Reading",
    "WholeGraphReading",
    "topology_only_energy",
    "topology_only_h1",
]

#: @type chosen
#: @flexibility free: the pair reading is ~3 ms on the real dome, so the number is set by how finely the fall is worth resolving rather than by cost
#: @warrant docs/adr/0010-restriction-map-scale-is-gauge-fixed.md
#: How often the paired per-edge instrument is read, in ticks.
#:
#: **Chosen here, not recorded.** The record fixes that these run "on the
#: diagnostic cadence, not per tick" (ADR-0010) and fixes no number. Ten is set
#: against the run the transport rule was measured on: 600 ticks on the real
#: dome took disagreement from 0.66 to 0.019 with an edge's joint scale grown
#: into `ρ = 2`, so the interesting stretch is the first hundred ticks or so and
#: a cadence of ten puts about ten readings inside it. The cost that buys is
#: nothing — a batched product over 1364 edge endpoints — so the number is set
#: by how finely the fall is worth resolving rather than by what it costs.
DEFAULT_EVERY = 10

#: @type chosen
#: @flexibility bounded by cost: one reading is a 3764x3764 eigendecomposition, ~16 s on the real dome
#: @warrant here
#: How often the whole-graph reading is taken, in ticks. Must be a multiple of
#: :data:`DEFAULT_EVERY`.
#:
#: **Chosen here, not recorded**, and this one is set by cost. One reading is a
#: `3764 × 3764` symmetric eigendecomposition, ~16 seconds on the real dome
#: against ~3 ms for the pair. At 100 that is six readings across the 600-tick
#: run above — enough to see `dim H⁰` and the minimum achievable energy move
#: with the maps, which is the whole of what makes them run-time measurements
#: rather than construction constants, and cheap enough that a run is not
#: dominated by its own instrumentation.
DEFAULT_WHOLE_GRAPH_EVERY = 100

#: @type chosen
#: @flexibility none that matters: the baseline is a generic rank, and tests/test_diagnostics.py holds two unrelated seeds to one number
#: @warrant here
#: The seed the topology-only baseline's generic maps are drawn at.
#:
#: The baseline is a **generic** rank, which is the same for almost every draw,
#: so the seed does not choose the answer — `tests/test_diagnostics.py` holds
#: that down by asking two unrelated seeds for the same number. It is fixed
#: anyway so that a recorded baseline is reproducible from the dome alone.
BASELINE_SEED = 0

#: `δ`'s sign on the two ends of an edge: `+` on `edge.u`, `−` on `edge.v`, the
#: same orientation :meth:`patchworks.tick.Sheaf.disagreement` reports in.
_SIGN = (1.0, -1.0)


class Condition(str, Enum):
    """What the world was doing while a reading was taken.

    ADR-0010's driven/quiescent contrast, which is the axis the paired
    instrument is read along and the same axis ADR-0007's quiescent hold
    separates the lag floor from the static floor on. Declared by whoever
    arranged the world; see this module's docstring for why it is not measured.
    """

    QUIESCENT = "quiescent"
    """The world is held still — `01-cell-and-sheaf.md`'s *at rest*. Energy
    draining here is the quiescent hold working: nothing is moving, so there is
    nothing for the slow end of an edge to be behind, and the lag floor drains
    (ADR-0007)."""

    DRIVEN = "driven"
    """The world is moving — `01-cell-and-sheaf.md`'s *while the world drives*.
    Energy draining here, with effective rank sliding toward 1 across the fleet,
    is the collapse this instrument exists to catch."""


@dataclass(frozen=True)
class EdgeReading:
    """The paired instrument: per-edge Dirichlet energy **and** per-edge effective rank.

    One object carrying both halves, because neither half means anything alone
    and this is where that is enforced rather than described. Both fields are
    required, and :meth:`__post_init__` refuses a pair whose two halves do not
    describe the same edges — a reading assembled from two different ticks, or
    from two different graphs, is the one way the pairing could be satisfied in
    the type and broken in fact.
    """

    energy: torch.Tensor
    """`[edges]`: the per-edge Dirichlet energy `‖F_u x_u − F_v x_v‖²`, one
    edge's own term of `xᵀLx`. This is the squared disagreement on that edge —
    the sheaf's only edge-level error signal, and the quantity over-smoothing
    makes vanish (`docs/spec/01-cell-and-sheaf.md`, *Disagreement, and what is
    done about it*)."""

    effective_rank: torch.Tensor
    """`[edges, 2]`: the participation ratio `(Σσᵢ²)² / Σσᵢ⁴` of each end's
    restriction map's singular values, reading 1 for a rank-1 map and `m` for a
    uniform one (ADR-0010).

    **Two columns, one per end, and they are not averaged.** An edge has two
    restriction maps belonging to two different cells, and effective rank is a
    property of a map. Column 0 is `edge.u`'s map and column 1 is `edge.v`'s, in
    the pair order :mod:`patchworks.restriction` fixes. Collapsing them to one
    number per edge would hide exactly the case the instrument is for — one end
    concentrating while the other does not — and "across the fleet" in the
    spec's reading is a statement about the whole population of maps, which is
    what this is."""

    def __post_init__(self) -> None:
        if self.energy.dim() != 1:
            raise ValueError(
                "energy is one value per edge, `[edges]`; got "
                f"{tuple(self.energy.shape)}"
            )
        edges = self.energy.shape[0]
        if self.effective_rank.shape != (edges, 2):
            raise ValueError(
                "the pair's two halves must describe the same edges: energy is "
                f"[{edges}] and effective rank is "
                f"{tuple(self.effective_rank.shape)}, which is not [{edges}, 2] "
                "(one column per end of each edge)"
            )
        for name, tensor in (
            ("energy", self.energy),
            ("effective_rank", self.effective_rank),
        ):
            if tensor.grad_fn is not None or tensor.requires_grad:
                raise ValueError(
                    f"{name} arrived on the autograd tape. A diagnostic is a "
                    "read of the sheaf, not a path back into it; see "
                    "docs/spec/09-the-build-stack.md, The locality guard."
                )

    def __len__(self) -> int:
        return int(self.energy.shape[0])


@dataclass(frozen=True)
class WholeGraphReading:
    """The measurements that are not construction constants, against the learned maps.

    `docs/spec/01-cell-and-sheaf.md`, *What is recorded, and what it is worth*:
    `χ` is fixed at construction and invariant under learning, but **`dim H⁰`**
    and the **minimum achievable Dirichlet energy under the world's boundary
    conditions** move as the restriction maps learn and have to be measured.

    **Over the predicting-cell subcomplex**, which is the record's own
    convention rather than a choice made here. `06-graph-topology.md` corrects
    `01-cell-and-sheaf.md` so that the `dim H⁰` bound and `χ` are computed over
    predicting cells only — including boundary cells "swamps the measurement
    outright", since the world overwrites their stalks every tick and a patch
    cell alone carries 48 nominally private components. Taking the same
    convention here makes the identity hold exactly:
    `dim H⁰ − dim H¹ = Σ_{v predicting} n − Σ_e m_e = χ`, which is
    :attr:`~patchworks.graph.Dome.euler_characteristic`. The edge term runs over
    **all** edges, boundary-incident ones included, for the reason that property
    gives.

    **That identity is a check on this convention and not on the assembly of
    `δ`**, and it is worth being exact about which, because the difference is
    easy to overclaim. Both dimensions are counted off one `rank`, so their
    difference is `Σ_v n − Σ_e m_e` for *any* rank at all and no arrangement of
    `δ` — no wrong sign, no shifted row, no dropped block — could break it. What
    it does catch is counting rows or columns by a different convention than
    `graph.py` counts `χ` by. `δ` itself is checked by building it a second time
    through another route: `tests/test_diagnostics.py` holds :attr:`rank`
    against an independently assembled coboundary and
    :attr:`minimum_energy` against a least-squares solve of the same system,
    and between them a dropped block, a flipped sign, a shifted row offset and a
    wrong column all move one of the two.
    """

    rank: int
    """`rank δ` over the predicting-cell subcomplex. Both dimensions below come
    from this one number, which is why they come from one decomposition."""

    dim_h0: int
    """`dim H⁰` against the learned maps: the predicting population's private
    dimension, measured rather than bounded. It is at least the construction
    bound `Σ_v max(0, n − Σ_{e∋v} m_e)`, which holds for any maps at all
    (`docs/spec/01-cell-and-sheaf.md`), and learned rank-deficiency only
    enlarges it."""

    dim_h1: int
    """`dim H¹` against the learned maps. Compared against
    :attr:`Diagnostics.h1_baseline`, not against zero: `H¹` has two sources —
    graph cycles and map rank-deficiency — and a lattice's cycle count
    guarantees plenty of irreducible disagreement whatever the maps do
    (`docs/spec/06-graph-topology.md`, *What the cycles do*). The excess over
    the baseline is the part attributable to the maps."""

    minimum_energy: float
    """The minimum achievable Dirichlet energy under the world's boundary
    conditions: `min_{x_P} Σ_e ‖δ_e (x_P, x_B)‖²` with every boundary cell's
    node stalk held at what the world wrote and every predicting cell's free.

    The floor no reconciliation could get below on this configuration with these
    maps. It is what the measured total energy should be read against — a total
    that has reached it is not a graph that has stopped disagreeing, it is a
    graph whose remaining disagreement is the part no node-stalk assignment can
    clear."""


@dataclass(frozen=True)
class Reading:
    """One cadence tick's record: the pair, the condition, and sometimes the whole graph.

    The condition is a field rather than something a caller remembers, because
    "the at-rest and under-drive readings are told apart in the record" is the
    whole of what makes the pair readable at all.
    """

    tick: int
    """The sheaf's tick counter when this was read."""

    condition: Condition
    """What the world was doing, as declared by whoever arranged it."""

    edges: EdgeReading
    """The paired per-edge instrument. Never absent, and never half of one."""

    whole_graph: WholeGraphReading | None = None
    """The whole-graph measurements, on the ticks the longer cadence lands on
    and `None` on the rest. When it is present it is a reading of **this**
    reading's configuration, which is what the multiple-of rule between the two
    cadences buys."""

    def __post_init__(self) -> None:
        # The condition is coerced here rather than only in `Diagnostics.read`,
        # because `report` groups by it with `is` and a `Reading` built by hand
        # is a public thing to build. `Condition` is a `str` enum, so a plain
        # `"driven"` compares equal to `Condition.DRIVEN` and is *not* it: such a
        # reading would sit in the record, be counted by neither group, and
        # vanish from the report — a reading silently absent from the record is
        # the one failure a diagnostic must not have.
        object.__setattr__(self, "condition", Condition(self.condition))


def topology_only_h1(dome: Dome, *, generator: torch.Generator | None = None) -> int:
    """`dim H¹` from the graph alone: the baseline `06-graph-topology.md` requires.

    ADR-0004 makes persistent structured irreducible disagreement a
    falsification signature — curvature the linear restriction map cannot
    follow. `06-graph-topology.md` records the cost that a lattice's enormous
    cycle count puts on reading it: if topology guarantees plenty of irreducible
    disagreement regardless, that attribution needs "a **topology-only
    baseline** to compare against rather than a comparison with zero". This is
    that baseline, and it is the number `dim H¹` is excess *over*.

    **What is held fixed and what is thrown away.** `H¹` has two sources: graph
    cycles, and map rank-deficiency, "which is invisible to graph topology and
    which Patchworks' masked maps guarantee"
    (`docs/spec/01-cell-and-sheaf.md`). Topology-only means the first source and
    not the second, so this is computed with **generic dense maps** of each
    edge's and cell's shapes — unmasked, unlearned, full rank almost surely.
    What is kept from the built graph is its edges, its `m_e` and its `n`; what
    is discarded is every reason a real map has for transmitting fewer
    directions than its shape allows.

    **A construction quantity.** It reads no learned parameter, so it moves only
    if the graph does — the same kind of object as `χ` or the reconciliation
    gain's denominator. :class:`Diagnostics` computes it once, at construction,
    and stores it.

    Over the predicting-cell subcomplex, for the reason
    :class:`WholeGraphReading` gives. `generator` seeds the generic draw; the
    answer is a generic rank and does not depend on it.
    """
    if generator is None:
        generator = torch.Generator().manual_seed(BASELINE_SEED)
    n = dome.shape.n
    row = {cell_id: i for i, cell_id in enumerate(dome.predicting)}
    rows = sum(edge.m for edge in dome.edges)
    columns = len(dome.predicting) * n
    delta = torch.zeros(rows, columns, dtype=torch.float64)
    at = 0
    for edge in dome.edges:
        for side, cell_id in enumerate((edge.u, edge.v)):
            if cell_id in row:
                block = torch.empty(edge.m, n, dtype=torch.float64).normal_(
                    generator=generator
                )
                column = row[cell_id] * n
                delta[at : at + edge.m, column : column + n] += _SIGN[side] * block
        at += edge.m
    return rows - _rank(delta)


#: @type chosen
#: @flexibility free: the level's noise falls as 1/sqrt(draws) and eight is already far below the factor the level is read at; the whole read is milliseconds
#: @warrant here
#: How many generic draws :func:`topology_only_energy` averages over.
#:
#: A generic *rank* is the same for almost every draw, which is why
#: :func:`topology_only_h1` needs one; a generic *energy* is not — one draw
#: gives one sample of `‖F_u x_u − F_v x_v‖²` and the per-edge spread across
#: draws is of the same order as the value. Eight is set by that: the standard
#: error of a mean falls as `1/√draws`, so eight puts the level's own noise
#: comfortably below the factor-of-two differences a reference level is read at,
#: and the whole thing is a batched product over edge endpoints costing
#: milliseconds.
TOPOLOGY_ENERGY_DRAWS = 8


def topology_only_energy(
    sheaf: Sheaf,
    *,
    draws: int = TOPOLOGY_ENERGY_DRAWS,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """`[edges]`: the disagreement generic full-rank maps would carry here.

    :func:`topology_only_h1`'s twin on the **energy** scale, and the reference
    level [#156](https://github.com/NGL321/patchworks/issues/156) left owed.
    That prototype used `06-graph-topology.md`'s topology-only baseline as a
    floor under a residual — *a measured construction quantity, and it is how
    this rule gets a reference level without inventing one* — and stood the
    level in at `0.05`, because no run had produced it. This produces it.

    **Same complex, same throwing-away.** Generic dense maps of each edge's and
    cell's shapes: what is kept from the built graph is its edges, its `m_e`,
    its stalk widths and its **scale**; what is discarded is every reason a real
    map has for transmitting fewer directions than its shape allows. The
    question it answers is *how much would this configuration disagree if the
    maps were generic and full rank* — so a residual no larger than it is
    disagreement the graph's own shape produces whatever any map does.

    **Scale is kept, and that is not a detail.**
    [ADR-0010](../adr/0010-restriction-map-scale-is-gauge-fixed.md) gauge-fixes
    every restriction map's Frobenius norm to `[1/ρ, ρ]`, pinned maps to exactly
    1, so unit Frobenius norm is the gauge's own centre — cited, not chosen. An
    unnormalised `normal_()` draw has `‖F‖_F ≈ √(m·n)`, which on a boundary edge
    is about 20, and a reference level two orders of magnitude off the scale of
    the thing it is a reference for is not a reference level. This still reads
    **no learned parameter**: the gauge is a construction fact, and the maps'
    own norms are not consulted.

    **Why this and not the null-space minimum.** The obvious energy twin is
    `‖P_null b‖²` — :meth:`Diagnostics.whole_graph`'s minimum achievable energy,
    computed with generic maps — and it is the wrong object *per edge*, for a
    reason worth recording rather than rediscovering. A predicting cell's stalk
    is free, so an edge with a predicting end contributes a column block to
    `δ_P`; generically the only rows outside `range(δ_P)` are the rows with **no
    free column at all**, which are exactly the boundary-to-boundary edges. The
    null-space minimum is therefore identically zero on every interior edge, and
    a per-edge level that is zero on three quarters of the graph divides badly.
    It remains the right whole-graph number and :meth:`Diagnostics.whole_graph`
    still reports it; it is not a per-edge one.

    Over the whole edge set, boundary-incident edges included, and in float64
    for the reason :func:`_rank` gives. `generator` seeds the draws.
    """
    if draws < 1:
        raise ValueError(f"draws is a count of generic draws, >= 1; got {draws!r}")
    if generator is None:
        generator = torch.Generator().manual_seed(BASELINE_SEED)
    dome = sheaf.dome
    with torch.no_grad():
        stalks = sheaf.stalks.detach().to(torch.float64)
    total = torch.zeros(len(dome.edges), dtype=torch.float64)
    for _ in range(draws):
        for edge in dome.edges:
            ends = []
            for cell_id in (edge.u, edge.v):
                width = dome.cells[cell_id].stalk
                block = torch.empty(edge.m, width, dtype=torch.float64).normal_(
                    generator=generator
                )
                # ADR-0010's gauge, as the scale of the generic draw. A map at
                # unit Frobenius norm is what a pinned map is exactly and what
                # every other map is within `ρ` of.
                block /= block.norm()
                ends.append(block @ stalks[sheaf.layout.slice(cell_id)])
            total[edge.id] += (ends[0] - ends[1]).pow(2).sum()
    return total / draws


def _rank(delta: torch.Tensor) -> int:
    """`rank δ`, through `δ δᵀ` rather than through `δ` itself.

    The coboundary is `[Σ_e m_e, Σ_v n]` — wide, and on the real dome that is
    `3764 × 4800`. Its Gram matrix is `3764 × 3764` and carries the same rank,
    and a symmetric eigendecomposition of it costs about a quarter of a full SVD
    of the original. The price is precision: the Gram squares the condition
    number, so a singular value below `√ε · σ_max` is not resolvable at all.
    That is why everything here runs in **float64** — at `√ε ≈ 1.5e-8` the
    unresolvable band sits far below the smallest non-zero singular value the
    real dome actually carries, and the standard tolerance below lands inside
    the gap rather than inside the spectrum.
    """
    eigenvalues = torch.linalg.eigvalsh(delta @ delta.T)
    return int((eigenvalues > _tolerance(eigenvalues, delta)).sum())


def _tolerance(eigenvalues: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """`λ_max · max(shape) · ε`: the textbook rank tolerance, on the Gram matrix."""
    return eigenvalues.max() * max(delta.shape) * torch.finfo(delta.dtype).eps


class Diagnostics:
    """The cadence, the baseline, and the readings a run leaves behind.

    Built on a :class:`~patchworks.tick.Sheaf` rather than on an
    :class:`~patchworks.agent.Agent`: everything read here is the graph's own
    state, so nothing in this module knows the sandbox exists and the whole
    instrument runs on a dome with no world attached.

    Driven from outside, and nothing holds a reference to one — the shape
    :mod:`patchworks.surface` and :mod:`patchworks.timescale` both take, for the
    same reason. The sheaf holds no diagnostics, and nothing a cell's
    computation is handed can reach one::

        diagnostics = Diagnostics(agent.sheaf)
        for _ in run(agent, 600, seed=0):
            diagnostics.observe(Condition.DRIVEN)

    Unlike the surface's recorder this does **not** have to see every tick.
    Everything it reads is a quantity of the tick it is read on rather than a
    difference between two, so a missed call costs a reading and never corrupts
    one. What a missed call does not do is move the next one: the cadence is
    counted on `sheaf.ticks` rather than on calls to :meth:`observe`, which is
    what makes the multiple-of rule below a statement about ticks and what makes
    two runs comparable at the same tick numbers. :meth:`observe` says what that
    costs a caller who ticks in strides of their own.
    """

    def __init__(
        self,
        sheaf: Sheaf,
        *,
        every: int = DEFAULT_EVERY,
        whole_graph_every: int = DEFAULT_WHOLE_GRAPH_EVERY,
        generator: torch.Generator | None = None,
    ) -> None:
        for name, value in (("every", every), ("whole_graph_every", whole_graph_every)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} is a cadence in ticks, >= 1; got {value!r}")
        if whole_graph_every % every:
            raise ValueError(
                f"whole_graph_every ({whole_graph_every}) must be a multiple of every "
                f"({every}), so that every whole-graph reading lands on a tick that "
                "also carries its paired per-edge reading of the same configuration. "
                "See patchworks.diagnostics, Two cadences."
            )
        self.sheaf = sheaf
        self.every = every
        self.whole_graph_every = whole_graph_every
        #: Every reading taken so far, in order. The record the two conditions
        #: are told apart in.
        self.readings: list[Reading] = []

        dome = sheaf.dome
        maps = sheaf.maps
        n = dome.shape.n
        self._rows = sum(edge.m for edge in dome.edges)
        self._columns = len(dome.predicting) * n
        row = {cell_id: i for i, cell_id in enumerate(dome.predicting)}
        # Where each predicting endpoint's map goes in the assembled coboundary,
        # and which endpoints are the boundary's. Both are read off the built
        # graph once: the shape of `δ` is a construction fact, and only the
        # numbers in it move.
        blocks: list[tuple[int, int, int, int, float]] = []
        boundary = torch.zeros(maps.pairs, dtype=maps.maps.dtype)
        at = 0
        for edge in dome.edges:
            for side, cell_id in enumerate((edge.u, edge.v)):
                pair = pair_index(edge.id, side)
                if cell_id in row:
                    blocks.append((pair, at, edge.m, row[cell_id] * n, _SIGN[side]))
                else:
                    boundary[pair] = 1.0
            at += edge.m
        self._blocks = tuple(blocks)
        self._boundary_pairs = boundary
        self._n = n

        #: `dim H¹` from the graph's cycles alone, computed at construction and
        #: stored for comparison (:func:`topology_only_h1`). Invariant under
        #: learning, because no learned parameter appears in it.
        #:
        #: **Constructing a `Diagnostics` therefore costs a decomposition**, of
        #: the same size as :meth:`whole_graph`'s — about 4 seconds and a
        #: transient `3764 × 4800` float64 matrix on the real dome. Eager rather
        #: than lazy because the baseline is a construction quantity and this is
        #: construction; the cost is named here so that a configuration sweep
        #: building one instrument per run knows it is paying it per run, and
        #: can build one instrument and re-point it instead.
        self.h1_baseline = topology_only_h1(dome, generator=generator)

        #: The dome everything above was read off. `self.sheaf` is a plain
        #: attribute and re-pointing it is a use the note above recommends, so
        #: the two have to be checked against each other before a reading --
        #: see :meth:`_on_the_construction_dome`.
        self.dome = dome

    def __repr__(self) -> str:
        return (
            f"Diagnostics(every={self.every}, "
            f"whole_graph_every={self.whole_graph_every}, "
            f"{len(self.readings)} readings)"
        )

    def _on_the_construction_dome(self) -> None:
        """Refuse a sheaf that is not the one the cached layout was read off.

        `__init__` reads the shape of `δ` off the built graph once — the row
        offsets, the column offsets, which endpoints are the boundary's, and the
        `H¹` baseline — because that shape is a construction fact and only the
        numbers in it move. Re-pointing :attr:`sheaf` at a sheaf on a *different*
        dome would leave `b` laid out by the new graph's edges while `δ` and the
        baseline came from the old one: either a shape error or, worse, a
        reading that is silently of neither graph.

        The same guard :mod:`patchworks.timescale`, :mod:`patchworks.tick` and
        :mod:`patchworks.agent` all carry, for the same reason. Re-pointing at
        another sheaf on the **same** dome is exactly the sweep the baseline's
        note recommends, and stays allowed.
        """
        if self.sheaf.dome is not self.dome:
            raise ValueError(
                "the diagnostics were built for a different dome; the cached "
                "coboundary layout and the H^1 baseline are that dome's. Build "
                "a Diagnostics per dome, and re-point `sheaf` only within one."
            )

    # -- the paired instrument ---------------------------------------------

    def edge_reading(self) -> EdgeReading:
        """Read per-edge Dirichlet energy and per-edge effective rank, together.

        The only way to obtain either. There is deliberately no
        ``edge_energy()`` and no ``effective_rank()`` beside it: one reading
        cannot tell collapse from a draining lag floor, and an API that handed
        one out would be an API that let a run be diagnosed from half the
        instrument.

        **Energy is recomputed from the node stalks as they now stand**, not
        read off :meth:`~patchworks.tick.Sheaf.disagreement`. That method reports
        what each end *broadcast* during the message-passing phase, which is the
        configuration as it stood before reconciliation edited it and before the
        world wrote the boundary cells — a perfectly good record of the tick, and
        the wrong thing to put beside :meth:`whole_graph`, whose minimum
        achievable energy is a property of the configuration that exists now.
        The two halves of a comparison have to be of the same configuration.

        **Effective rank is computed from the map's Gram matrix, not from an
        SVD**, and the two are the same number rather than an approximation of
        it: `Σσᵢ² = ‖F‖_F²` and `Σσᵢ⁴ = ‖F Fᵀ‖_F²`, so the participation ratio
        is `‖F‖_F⁴ / ‖F Fᵀ‖_F²` exactly. ADR-0010 sizes the cost as "one small
        SVD per map at `m ≈ 4`"; this is that quantity, taken by the cheaper
        identity, and `tests/test_diagnostics.py` holds it against an explicit
        `svdvals` computation. The padded rows and masked columns of the stored
        map tensor contribute zero singular values, which change neither sum, so
        the whole population is one batched product.
        """
        self._on_the_construction_dome()
        sheaf = self.sheaf
        maps = sheaf.maps
        with torch.no_grad():
            outgoing = maps.restrict(sheaf.stalks[sheaf.layout.pair_positions])
            ends = outgoing.reshape(-1, 2, maps.edge_width)
            energy = (ends[:, 0] - ends[:, 1]).pow(2).sum(-1)

            # `‖F‖_F⁴ / ‖F Fᵀ‖_F²` is exactly scale-invariant, so it is taken
            # on the *unit-normalised* map, where that invariance is structural
            # instead of arithmetical. For `G = F/‖F‖_F` the ratio is plainly
            # `1 / ‖G Gᵀ‖_F²`, and that denominator lies in `[1/m, 1]` for any
            # `G` at all -- so the quotient cannot overflow, underflow, or need
            # an epsilon holding it up. Taken on `F` directly it would need one:
            # `‖F‖_F⁴` underflows to zero at norms an unprojected map tensor can
            # really reach, and **an absolute floor under a scale-invariant
            # quantity is a false collapse reading** -- a fleet at `‖F‖_F = 1e-8`
            # would report an effective rank near zero while genuinely
            # transmitting four directions, which is exactly the reading this
            # instrument exists to be unable to produce by accident.
            # In float64 for the same reason `_rank` is: the normalisation
            # squares entries on its way to a norm, so a float32 `‖F‖_F²`
            # overflows around `1e19` and a scale-invariant quantity comes back
            # `inf`. float64 moves that ceiling past any magnitude a map tensor
            # can hold, at the cost of one small cast.
            weights = maps.maps.detach().to(torch.float64)
            norms = weights.flatten(1).norm(dim=-1)
            transmitting = norms > 0
            safe = torch.where(transmitting, norms, torch.ones_like(norms))
            unit = weights / safe.view(-1, 1, 1)
            gram = torch.bmm(unit, unit.transpose(1, 2))
            # A map transmitting nothing reads `0`, which is *below* the
            # degenerate limit of 1 rather than clamped up to it. Nothing here
            # is a rank floor: no map's reading is ever raised.
            effective = torch.where(
                transmitting,
                1.0 / gram.flatten(1).pow(2).sum(-1),
                torch.zeros_like(norms),
            ).to(maps.maps.dtype)
        return EdgeReading(
            energy=energy.detach().clone(),
            effective_rank=effective.detach().reshape(-1, 2).clone(),
        )

    # -- the whole-graph measurements --------------------------------------

    def whole_graph(self) -> WholeGraphReading:
        """`dim H⁰`, `dim H¹` and the minimum achievable Dirichlet energy.

        One symmetric eigendecomposition of `δ_P δ_Pᵀ` answers all three, which
        is why they are one method and one reading rather than three. `δ_P` is
        the coboundary restricted to the predicting cells' node stalk
        components; the boundary cells' components are not free — the world
        writes them and they are the *boundary conditions* the minimum is taken
        under — so their contribution is a fixed vector `b` on the right-hand
        side rather than a column block.

        From the decomposition `δ_P δ_Pᵀ = V Λ Vᵀ`:

        * `rank δ_P` is how many eigenvalues clear the tolerance, and the two
          dimensions follow by counting;
        * the minimum achievable energy is `‖P_null b‖² = Σ_{λ ≤ tol} (vᵀb)²`,
          the part of the boundary's contribution that lies outside anything the
          predicting cells can produce.

        That form is used rather than the equivalent `‖b‖² − ‖P_range b‖²`
        because the second is a small difference of two large numbers and loses
        most of its significant figures on the real dome; the first is a sum of
        squares and loses none. `tests/test_diagnostics.py` checks it against a
        least-squares solve of the same system.

        Runs in float64 throughout, for the reason :func:`_rank` gives.
        """
        self._on_the_construction_dome()
        sheaf = self.sheaf
        maps = sheaf.maps
        with torch.no_grad():
            delta = torch.zeros(self._rows, self._columns, dtype=torch.float64)
            weights = maps.maps.detach().to(torch.float64)
            for pair, at, m, column, sign in self._blocks:
                delta[at : at + m, column : column + self._n] += (
                    sign * weights[pair, :m, : self._n]
                )

            # The boundary's fixed contribution, in the same row layout: what
            # the boundary ends put on their edges, with the predicting ends
            # zeroed out, differenced by edge and laid out end to end.
            #
            # Restricted here in float64 off the `weights` already built, rather
            # than through `maps.restrict`, which would work in the sheaf's own
            # float32. `b` is not a bystander in the precision argument above:
            # the minimum is `‖P_null b‖²`, so a float32 `b` puts a ~1e-7
            # relative cap on the floor and reimposes most of the loss the
            # null-space form was chosen to avoid -- hardest exactly where the
            # floor is small against `‖b‖²`, which is the regime that argument
            # is about.
            stalks = sheaf.stalks[sheaf.layout.pair_positions].to(torch.float64)
            outgoing = torch.bmm(weights, stalks.unsqueeze(-1)).squeeze(-1)
            held = (
                outgoing * self._boundary_pairs.to(torch.float64).unsqueeze(-1)
            ).reshape(-1, 2, maps.edge_width)
            by_edge = held[:, 0] - held[:, 1]
            b = torch.zeros(self._rows, dtype=torch.float64)
            at = 0
            for edge in sheaf.dome.edges:
                b[at : at + edge.m] = by_edge[edge.id, : edge.m]
                at += edge.m

            eigenvalues, vectors = torch.linalg.eigh(delta @ delta.T)
            null = eigenvalues <= _tolerance(eigenvalues, delta)
            rank = int((~null).sum())
            minimum = float((vectors[:, null].T @ b).pow(2).sum())

        return WholeGraphReading(
            rank=rank,
            dim_h0=self._columns - rank,
            dim_h1=self._rows - rank,
            minimum_energy=minimum,
        )

    # -- the cadence -------------------------------------------------------

    def read(
        self, condition: Condition | str, *, whole_graph: bool | None = None
    ) -> Reading:
        """Take a reading now, whatever the cadence says, and record it.

        `condition` is required — see this module's docstring for why it is
        declared rather than measured.

        `whole_graph` decides the expensive half. Left `None` it follows the
        longer cadence, which is what :meth:`observe` wants; passed `True` or
        `False` it overrides, which is what **the quiescent hold** wants.
        ADR-0007's protocol is to hold the world still, sweep configurations and
        watch what drains, and a sweep step lands on whatever tick it lands on.
        The minimum achievable energy is the number that hold is read against —
        a total that has stopped falling has either drained to the floor or
        stopped for another reason, and only the floor tells them apart — so an
        instrument that could not produce it off the grid would be missing the
        reading at exactly the moment the protocol calls for it.

        The override is on :meth:`read` and not on :meth:`observe` deliberately.
        `observe` is the cadence, and the multiple-of rule between the two
        cadences is what guarantees its whole-graph readings land on ticks that
        also carry a paired per-edge reading of the same configuration. `read`
        is the caller saying which reading they want; forcing the pair is not
        possible there either way, because the pair is never optional.
        """
        condition = Condition(condition)
        tick = self.sheaf.ticks
        if whole_graph is None:
            whole_graph = tick % self.whole_graph_every == 0
        reading = Reading(
            tick=tick,
            condition=condition,
            edges=self.edge_reading(),
            whole_graph=self.whole_graph() if whole_graph else None,
        )
        self.readings.append(reading)
        return reading

    def observe(self, condition: Condition | str) -> Reading | None:
        """Read the tick that just finished, on the cadence. Returns it, or `None`.

        `None` on the ticks the cadence skips, which is most of them — that is
        the point of the cadence and of this ticket. Nothing here is a
        difference between two consecutive ticks, so a skipped call costs a
        reading and never corrupts one.

        **The cadence is counted on the sheaf's own tick counter, not on calls
        to this method**, and that is worth being plain about because the two
        come apart. Readings land on ticks divisible by :attr:`every`, so two
        runs are comparable at the same tick numbers whoever was calling and
        however often — and the multiple-of rule between the two cadences means
        anything at all, since it is a statement about tick numbers. The price
        is that a caller who ticks in strides of their own and calls once per
        stride is not decimated but **filtered**: strides of ten from an odd
        starting tick never land on a multiple of ten, and such a caller records
        nothing at all rather than one reading per stride. :meth:`read` is the
        way to take a reading at a tick of the caller's choosing.

        Tick 0 is a multiple of everything, so a call made before the first tick
        reads the initial configuration and, at the default cadences, pays the
        whole-graph decomposition on it. On a freshly built sheaf that is an
        all-zero configuration: every edge agrees exactly, the boundary
        conditions assert nothing, and the energies and the floor are all zero.
        It is a true reading of a real configuration and a useless baseline to
        put a run's later readings against. Driving this from `run()` never sees
        it, because a tick has finished by then.
        """
        if self.sheaf.ticks % self.every:
            return None
        return self.read(condition)

    # -- reading the record ------------------------------------------------

    def report(self) -> str:
        """The readings, grouped by the condition they were taken under.

        Grouped that way because that grouping *is* the diagnosis: the same fall
        in energy means opposite things under the two conditions, and a report
        that pooled them would be the one-reading instrument this module exists
        to refuse.
        """
        lines = ["Diagnostics", "===========", ""]
        lines.append(f"topology-only dim H^1 baseline: {self.h1_baseline}")
        lines.append(
            "  what the graph's cycles guarantee with no map rank-deficiency at all"
        )
        lines.append("")
        lines.append(
            f"cadence: every {self.every} ticks, whole graph every "
            f"{self.whole_graph_every}"
        )
        lines.append("")

        if not self.readings:
            lines.append("no readings yet")
            return "\n".join(lines)

        for condition in Condition:
            taken = [r for r in self.readings if r.condition is condition]
            count = len(taken)
            lines.append(
                f"{condition.value} ({count} reading{'' if count == 1 else 's'})"
            )
            if not taken:
                lines.append("  none")
                lines.append("")
                continue
            first, last = taken[0], taken[-1]
            lines.append(f"  ticks {first.tick} -> {last.tick}")
            for label, pick in (
                ("per-edge Dirichlet energy, mean", lambda r: r.edges.energy),
                ("per-edge effective rank, mean", lambda r: r.edges.effective_rank),
            ):
                lines.append(
                    f"  {label:<32} {float(pick(first).mean()):.6g} -> "
                    f"{float(pick(last).mean()):.6g}"
                )
            lines.append(
                f"  {'per-edge effective rank, min':<32} "
                f"{float(first.edges.effective_rank.min()):.6g} -> "
                f"{float(last.edges.effective_rank.min()):.6g}"
            )
            lines.append("")

        whole = [r for r in self.readings if r.whole_graph is not None]
        if whole:
            latest = whole[-1]
            measured = latest.whole_graph
            assert measured is not None  # for the type checker; `whole` filtered on it
            bound = int(self.sheaf.dome.private_dimensions.sum())
            lines.append(
                f"whole graph, most recent (tick {latest.tick}, "
                f"{latest.condition.value})"
            )
            lines.append(
                f"  dim H^0 = {measured.dim_h0}, against the construction bound "
                f"{bound}"
            )
            lines.append(
                f"  dim H^1 = {measured.dim_h1}, against the topology-only baseline "
                f"{self.h1_baseline} "
                f"({measured.dim_h1 - self.h1_baseline:+d} from the maps)"
            )
            lines.append(
                f"  dim H^0 - dim H^1 = "
                f"{measured.dim_h0 - measured.dim_h1} = chi "
                f"({self.dome.euler_characteristic}), the convention holding"
            )
            lines.append(
                "  minimum achievable Dirichlet energy under the world's boundary "
                f"conditions: {measured.minimum_energy:.6g}"
            )
            lines.append(
                f"  measured total: {float(latest.edges.energy.sum()):.6g}"
            )
        return "\n".join(lines)
